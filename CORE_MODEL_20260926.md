# Neural core selection and training

The task is to identify the best features of our models from September 12–26,
create a coherent core, train it, and submit the resulting neural policy.
The earlier public-script submissions were outside that intended objective.
They are not candidates in this campaign.

## Thirty-minute run cap (supersedes the budgets below)

No job runs longer than thirty minutes. `scripts/queue_core_campaign.py` now
gives PPO a 27-minute trainer budget (`--max-hours 0.45`) in place of the
500-actor-wave budget, which took about 100 minutes. Arms therefore end at
different wave counts (roughly 100–130 actor waves at the observed rate), and
comparisons use the exact waves both arms reached (25/50/100). The panel cull's
150-wave warmup is not reached inside that budget; the cap bounds each run instead.
The eight-hour and four-hour limits recorded below are historical.

## PPO recipe from the September 27 ablations (supersedes the PPO settings below)

Three staged, single-factor campaigns fine-tuned the eight-epoch WDL clone under the cap:
- stage one: `artifacts/probes/ppo-ablations-20260927`;
- stage two: `artifacts/probes/ppo-stage2-20260927`;
- stage three: `artifacts/probes/ppo-stage3-20260927`.

Each arm was scored at exact waves 50 and 100 and at its endpoint:
- 512 games per opponent;
- starter, V27 and the clone itself, each argmax and sampled;
- the same seeds throughout, and the references reproduce their scores exactly.

The clone is the reference that matters. Its argmax already beats V27 97% of the time, so PPO only helps if it beats the clone.

Findings:
- **Actor GAE λ 1 is decisive.**
  - At λ 0.972, every arm collapses after the clone: zero argmax V27 wins at wave 50.
  - At λ 1, the policy beats the clone 99% argmax and 93% sampled at wave 50, and raises sampled V27 from 0.50 to 0.87.
  - Truncated credit over a weak critic (R² 0.1–0.2) was what destroyed the clone.
- **Full-rate λ 1 drifts after wave 50.** Against the clone it falls to 0.68 argmax by the endpoint, while V27 stays at 1.00. Scripted opponents and the panel cannot see this; the clone reference can.
- **Actor LR 5e-5 holds the peak.** Its endpoint scores 0.936 argmax against the clone and 1.000 against V27.
- **Keep the JEPA objective.** Without it, the slow rate drifts too (0.714 at the endpoint). At the full rate, dropping it is neutral.
- **League-heavy (64 self-play + 128 league) hurts:** sampled V27 falls to 0.52 at wave 100.
- **Minibatch 16384 does not fit on the device.**

`scripts/queue_core_campaign.py` now uses actor GAE λ 1 and actor LR 5e-5, with retained update activations (`../ppo-speed-20260927`: the same function, 0.36 s faster per wave).

The adopted arm, stage two's `lambda-1-actor-lr-5e-5`, is also the repository default:
- a `train_ppo.py` launch that states only `--run-dir`, `--init-actor-from`, `--iterations`, `--max-hours` and `--seed` resolves to exactly its command, model configuration included;
- `production.build_training_command` emits it in full;
- the core campaign still states every flag, because it launches frozen source snapshots whose defaults predate this.

The strongest single checkpoint measured is `runs/ppo-ablations-20260927/lambda-1/checkpoint-000060.pt`, which is λ 1 at the full rate, wave 50.

Every conclusion rests on one seed per arm. Differences smaller than about ±0.06 are ties.

## Executing step-based comparison (supersedes earlier queue recipes)

Authoritative manifest: `artifacts/probes/core-model-20260926/campaign-step.json`.
Frozen source: `ada2ea7bc676e51ab92595290a3a95076f31190e1c0fc60d85c706d85b5deead`.
Runs live under `runs/core-step-model-20260926`.

The trainer now enforces 500 actor-active waves using its persisted panel counter,
with a 540-total-iteration safety ceiling, no internal wall-time cutoff, and a
checkpoint every 25 active waves. Warmup does not consume the actor-wave budget.
KL can reduce actual optimizer updates, so those are recorded separately and
must be checked when interpreting matched exposure. A cull or operational timeout
leaves later milestones incomplete; it does not supply a comparable endpoint.

Panel version 2 includes starter, V27 and V16. Its selection score uses argmax
only; sampled score and live-policy critic calibration remain separate. The
posttraining evaluator compares exact waves 25/50/100/200/300/400/500, 512 games
per opponent in each decoding mode, with hashes and cumulative update/state
accounting. No nearest/latest substitution is allowed. Cross-model official
engine evaluation remains a subsequent stage.

Jobs 10253–10256 clone the four feature-chain arms (20-minute limits), 10257–10258
perform BC admission panels (25-minute limits), 10259–10262 perform the four PPO
runs (eight-hour operational limits), and 10263 evaluates matched steps (four-hour
limit). All are exclusive, default-priority, one-attempt MLQ jobs. Superseded
10244–10252 were canceled or dependency-skipped before execution. Completed old
BC runs are preserved, but the short clone stage is rerun under one common source
identity. Trainer/panel tests: 127 passed; the step selector and admission tests
are checked separately.

The second-stage auxiliary-loss control is implemented as `--jepa-loss-scale 0`
in frozen source `de529aa093043727716746159676be7576db6fd6a50ccbb07bc2a08add04354b`.
It preserves the backbone optimizer, policy gradients, LR, clipping and KL gate,
while holding objective-only weights and moments. Default scale 1 preserves the
original graph. Raw auxiliary computation remains for this loss-removal control;
compiled CUDA execution is not yet verified. It will start from the same selected
initializer as its control, not from a longer-trained policy. Setting the older
individual coefficients to zero is still invalid for LeJEPA optimizer ownership.

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

## Functional collapse evidence

B2 release wave 15 to wave 27: self-play planting fraction remained about 3%,
but harvest actions fell 3.97% to 1.45%, selling 10.15% to 5.80%, and passing
rose 21.17% to 31.66%. S1 harvest stayed 4.52% to 4.70%, selling 12.18% to
12.45%, and passing 17.05% to 16.62%. B1 began with only 1.32% harvest and
4.79% selling while the actor was frozen. These are action-frequency observations
from different state distributions, not causal attribution or per-opportunity
success rates. They motivate checking sequence completion and resource use
around actor release, alongside policy/backbone/critic diagnostics.

Raw selected measurements: `artifacts/probes/core-model-20260926/collapse-function-metrics.json`.

## Paused for parameter-golf priority

User stopped job 10260 after about 31 minutes and deferred further experiments.
Canceled queued jobs 10261–10263. Schema3/ALL/no-affordance is the leading
candidate; checkpoint 91 (75 actor waves) scored argmax 64/64 starter, 64/64
V27, 60/64 V16 on the development panel. Latest durable checkpoint is 116
(100 actor waves); logging reached 140 (124 actor waves), so resuming replays
24 unsaved waves. Best/latest checkpoints and the full stop-time journal are
preserved under `artifacts/probes/core-model-20260926/paused-schema3-all`;
`resume.json` records hashes and the resume command. No remaining GPU work is
queued for this campaign. Final validation and submission remain deferred.

## Post-pause implementation and reward analysis

Training remains paused. New LeJEPA configurations retain schema 4, which already
includes the signed-log money difference (for nonnegative banks,
`log((own + 1) / (opponent + 1))`) alongside absolute features. The paused arm
explicitly used schema 3. Reward prediction is zero by default, including the
core campaign recipe; this is separate from changing the terminal RL reward.

New configurations enable destination-based movement and exact pre-order market
resource conditioning. Movement scores own-farm destinations and marginalizes
their probability into legal next steps along Manhattan routes. It reconsiders
the destination every turn, without a persistent route commitment. BC and PPO
score the executed direction, summing all destination aliases. Market kind and
quantity preferences receive zero-initialized linear residuals from post-unit,
pre-order cash, shed stocks, market inventory, hires, land and seed holdings.
Recorded prefix features are replayed in BC/PPO; native and Python decoding use
the same feature ordering. Integer quantities and ALL marginalization remain.

The saved-config decoder leaves both new heads disabled when their flags are
absent. This does not migrate unrelated obsolete configuration fields: the
paused checkpoint still requires its preserved source snapshot. Its weights
have not been changed or trained with these additions.

`scripts/compare_terminal_rewards.py` rescored saved evaluation outcomes and
explicit synthetic lotteries without running a model or game. Results are in
`artifacts/probes/core-model-20260926/reward-proxy-comparison.json`. Current main
defaults to hard `terminal-outcome`; the paused snapshot used
`terminal-soft-outcome`. For `L = log((own + 3000)/(opponent + 3000))`, the paused
reward is exactly `tanh(tanh(L/2)/0.01)`. A raw log ratio retains large-margin
information, but changes expected-policy preferences: the explicit 60%-large-win
lottery beats the 90%-modest-win lottery under log utility. Rescoring cannot
predict learning under a new reward. Keep the terminal objective unchanged;
raw log rewards would also require recalibrating value support and loss scales.

The named cleanrl concat run detaches normalized JEPA features before the actor,
concatenates raw observations, and uses an independent raw-observation critic.
Its JEPA prediction target detaches after projection; SIGReg still trains both
projected populations. Separate PPO/SSL optimizers, repeated SSL exposures,
normalized latent scaling and normalized residual readouts are substantial parts
of that recipe. Its residual readouts are not additive raw/latent fusion.
Matched 7–9M step windows do not establish a consistent concat advantage over
the latent-only sibling; the sibling's completed late returns exceed 12k.
Training episodic returns from one seed are not held-out evaluation evidence.
Do not equate this topology with simply detaching this game's shared backbone:
the existing game default still admits policy gradients and excludes critic
gradients. A fully independent JEPA representation with raw task-trained actor
and critic paths is a future controlled alternative, not a demonstrated win here.

## Elo-aligned critic comparison

The objective is match score, `P(win) + 0.5 * P(draw)`, against the intended
opponent population. Money margin is diagnostic, not a selection objective.
New campaign commands explicitly use hard terminal outcomes. Training remains
paused; the prior snapshot's soft-reward recipe is preserved with that snapshot.

LeJEPA's default critic head is now WDL (`--wdl-value true`): a loss/draw/win
classifier trained by unsmoothed cross entropy, with PPO value
`P(win) - P(loss)`. It requires `--reward-mode terminal-outcome --gamma 1
--critic-gae-lambda 1` (the trainer defaults); validation
rejects intermediate rewards, soft labels, and discontinuous trajectory masks.
Exact zero-baseline Monte Carlo returns supply labels, avoiding cancellation
residue from lambda-one arithmetic around nonzero value estimates. The head is
critic-only, so the same BC actor can initialize both comparison arms. HL-Gauss
remains available with `--wdl-value false`, which other reward modes and the
scalar head require. Saved LeJEPA configurations without the field restore as
HL-Gauss. WDL became the default by decision, before the matched comparison
(jobs 10369–10371) established better Elo.

Both critics report pre-update, state-weighted expected-match-score MSE, bias,
10-bin calibration error, and out-of-range prediction frequency under hard
outcomes with gamma one. These do not measure three-class draw calibration or
held-out strength. Compare critic arms at matched steps and select by held-out
match results/Elo, rather than cross-entropy magnitude or calibration alone.

Potential-based shaping is deferred: no extra reward mechanism is justified
before testing the existing outcome objective and critic. A learned potential
duplicating PPO's own value baseline need not add useful information.

## Frontier yardstick: demand-advance4 (2026-09-28)

V27 and the BC clone are saturated, so progress is now measured as own bank and
margin against demand-advance4 on paired seeds
(`artifacts/probes/ppo-frontier-20260928`). Every checkpoint so far, from the
v16 clone through thirty-minute PPO under the default recipe, loses all games
to it by about 35k with an own bank of about 77k. Neither an own-bank
group-relative advantage, a native demand-advance4 league lane, map-seed
grouping, nor cloning demand-advance4 itself moved that. At 0.002 nats per
decision the policy samples almost nothing outside the clone's strategy, so
exploration, not the reward, is the binding constraint.
