# Concurrent Population League Plan

## Decision

Training play becomes a **population of N agents that all learn at the same time**.
Every game pairs two *distinct* members of that population. There is no mirror
self-play, no frozen historical snapshot, no `scripted-v27` lane, and no built-in
agent inside the training wave. All members share one architecture and one
configuration; each owns its own actor weights, its own critic, and its own
optimizer state.

`N = 4` by default: each agent faces the other three. (Read literally, "one of 4
opponents" could mean four opponents *besides* the learner; that is `N = 5` and
the only change is the constant. `--population` carries it, so the plan does not
depend on which was meant.)

This is AlphaStar-like in the one sense that matters here — a population of
concurrent learners rather than a single learner against its own past — and
deliberately unlike AlphaStar in another: AlphaStar kept past players precisely
to suppress cycling. That risk is retained knowingly and instrumented in
Stage 5 rather than answered with frozen lanes.

## What this replaces, and the measurement that condemns it

Run `runs/ppo-v27-league`, 236 iterations, 75,520 learner trajectories:

| opponent | trajectories | share | score rate |
|---|---|---|---|
| itself, current weights (mirror) | 52,864 | 70.0% | 0.500 exactly, all 236 rows |
| its own frozen ancestors | 17,340 | 23.0% | 0.645 |
| `scripted-v27` | 3,115 | 4.1% | 0.005 |
| `starter` / `pass` / `random` | 2,201 | 2.9% | 0.76 - 0.98 |

Three separate defects, all removed by the population scheme:

1. **70% of experience carried no gradient signal.** `_relative_score(a, b) =
   (a - b) / (a + b)` (`src/kaggriculture/encoding.py:373`) is zero whenever
   `a == b`. `self_play_score_rate` took exactly one distinct value, 0.500, in
   every one of 236 rows — whether both banks held 150,000 or 3,000.
2. **The frozen-ancestor lanes rewarded destruction.** The ladder did include
   competent ancestors: iteration 0 (the BC clone) was a lane 27 times, and 383
   of 1,224 lane appearances were pre-collapse checkpoints. The learner scored
   0.645 against them while its own bank fell 5x, because market denial wins:
   measured head-to-head, iteration-49 weights against `bc5` bank 14,427 mean
   against `bc5`'s 6,477 (median 0) for a 0.69 win rate, while `bc5` against a
   copy of itself banks 39,801.
3. **The one honest lane was 4.1% of the wave and saturated.** At 0 versus
   186,576 the reward is pinned at -1.0, and a constant reward has no advantage.

The end state was a deterministic replay at iteration 236 banking **0** against
`public-v27`'s 186,576 while `league_score_rate` read 0.578 and rising.

## Objectives

- Every trajectory the learner collects must be able to change the objective:
  no pairing whose reward is zero by construction.
- Opponents must improve as the learner improves, without any frozen artifact,
  so the wave never spends budget on weaker copies of the same policy.
- Preserve exact masked categorical likelihoods and update-replay parity: the
  behaviour policy that sampled an action must be the one the update replays.
- Cost no more per iteration than the mixed wave it replaces, at equal total
  trajectories; the population must not buy diversity with wall clock.
- Keep one absolute, out-of-distribution measurement of strength — outside the
  training wave, in evaluation only.

## Non-objectives

- No frozen snapshot lanes, PFSP retirement contests, or league admission in the
  training path. That machinery is retired, not kept in parallel.
- No scripted or built-in opponent in the training wave. The native built-ins
  stay, used by evaluation and parity audits only.
- No architecture change. `entity-cnn` and its production `ModelConfig` are held
  fixed so the measured difference is the play scheme.
- No reward change bundled into this work. The saturation risk is documented
  below and decided separately; bundling it would confound the ablation.

## Design

### The wave is one vmapped forward, not N forwards

The infrastructure already exists and needs generalizing, not writing.
`_StackedFrozenEnsemble` (`src/kaggriculture/rollout.py:724-825`) stacks several
actors' parameters lane-wise and runs **one** `torch.vmap` over
`torch.func.functional_call`, with lanes padded to a common width
(`rollout.py:1156-1181`) so a captured CUDA graph survives a changing mix. It was
built for frozen opponents; nothing in it requires the weights be frozen, because
rollout takes no gradient — PPO replays stored actions in the update.

So the population wave is *simpler* than today's mixed wave: today runs two
forwards per step (the learner over its 320 rows, plus the ensemble over ~98
frozen rows). The population runs **one** ensemble forward of N lanes covering
every row, with lane index = agent index. Same total width, one launch sequence
instead of two.

The stacked tensors are refilled in place each iteration from live weights via
the existing `load()` path (`rollout.py:758-766`), which already exists to reload
snapshots and does not care that the source is now a learner.

### The native engine needs no new capability

`rust/kagg_env/src/python.rs:290-406` already accepts a stack of quantity heads
shaped `[heads, MARKET_KINDS, rank]` together with a per-row `head_ids: u16`.
Today that stack is `(actor, *opponents)` (`rollout.py:1119`) and `head_ids` is
0 for learner rows. For a population it becomes all N agents' heads with
`head_ids` in `0..N-1`. Per-row temperature and determinism flags are already
per-row arrays. Expected Rust diff: none. Stage 0 verifies this rather than
assuming it.

### Pairing schedule

With N = 4 there are `N(N-1) = 12` ordered (seat 0, seat 1) pairings. A wave of
`G` games assigns `G/12` games to each, so every agent plays every opponent
equally often on both seats and seat bias cancels exactly. `G` must be a multiple
of 12.

Uniform round-robin is the Stage 3 default because with four co-equal agents PFSP
has almost nothing to weight; PFSP over the three opponents by recent
head-to-head is a Stage 6 ablation, using the weighting already in `league.py`.

### Data budget arithmetic

Each game yields two learner trajectories now (both seats belong to learners),
where a league game yielded one. Per agent, trajectories per iteration are
`2G/N`.

| wave | G | total trajectories | per agent | expected cost |
|---|---|---|---|---|
| today's mixed wave | 112 self-play + 96 league | 320 | 320 | 15.1 - 15.6 s/iter |
| cost-parity population | 156 | 312 | 78 | one forward instead of two, at equal width |
| data-parity population | 636 | 1,272 | 318 | 4x batch, sub-4x time |

The second row holds wall clock and cuts per-agent data 4x; the third holds
per-agent data and pays in batch. Batch is the cheaper axis here: the rollout
forward is launch-gap bound, 859 kernel launches summing 4.9 ms inside a measured
12.94 ms. Stage 0 measures both rows before Stage 5 picks one; nothing in the
plan assumes the answer.

The **update** cost is unchanged at equal total trajectories: today 320
trajectories x 719 steps at `minibatch_size=2048` is 112 minibatches; four agents
of 318 trajectories are 4 x 111. Per-agent memory is one extra actor, critic, and
NorMuon state — about 2.8M parameters and their moments per agent, negligible
against 32 GiB.

### Per-agent updates, not a pooled one

`RolloutBatch` (`rollout.py:58-79`) gains one field, `agents`, alongside `seats`.
The update partitions on it and runs `update_ppo` once per agent. This is not
cosmetic: `prepare_advantages` normalizes by the batch's own advantage standard
deviation, so a pooled batch would normalize each agent's advantages by the
population's spread and leak one agent's return scale into another's step size.
Partitioning first makes per-agent normalization automatic.

Critic warmup (`--critic-warmup-iterations`) runs per agent, concurrently, at no
extra wall clock.

### Initialization: same competence, different weights

Four agents from one BC checkpoint are numerically identical, and their first
games are mirrors with reward 0 by symmetry. Instead: four BC runs on the same
mixed corpora with different seeds, under the new pretraining recipe (NorMuon
plus nanogpt's weight-decay treatment). Equal competence, genuinely different
weights, measured before launch.

### Strength measurement moves entirely outside training

With no built-in and no `scripted-v27` lane, nothing inside the wave measures
absolute strength — by design, since every internal number is relative. The
existing external evaluator (`scripts/external_eval_worker.py`,
`--external-eval-every`) becomes the only absolute signal and must run for **all
N agents** against `starter`, `pass`, `random`, and `public-v27`. It is CPU-only,
so it does not contend with the GPU.

Submission selection is unchanged in kind and N times wider in candidates:
`scripts/select_checkpoint.py` screens on a fixed panel, `scripts/build_submission.py`
gates on `--minimum-score-rate` and `--minimum-builtin-score-rate`.

## Stages

Each stage is a commit with its own evidence. No stage is skipped on the grounds
that the next one subsumes it.

**Stage 0 - verify the three load-bearing assumptions.** (a) The native sampler
accepts `head_ids` spanning N heads with no Rust change: assert a 4-head stack
reproduces four separate single-head calls exactly. (b) A vmapped ensemble
forward matches the plain module forward closely enough for replay parity: run
`scripts/audit_replay_parity.py` against the existing ceilings, since the
behaviour policy is now produced by `vmap` + `functional_call` while the update
uses the plain module. (c) Measured cost of one N-lane ensemble forward at both
wave sizes versus today's two forwards, using `scripts/sweep_rollout_execution.py`.
*Acceptance:* (a) bit-exact or documented tolerance, (b) parity within the
shipped ceilings, (c) a table of ms/step for both candidate wave sizes.

**Stage 1 - four diverse initializations.** Four BC runs, same corpora, different
seeds, new pretraining recipe. The corpora are the **`public-v16`** set, not the
v27 set the earlier clones used: measured head-to-head in the official engine
over three seeds and both seat orders, `public-v16` beat `public-v27` 6/6 with
median bank 77,261 against 59,489, so it is the stronger teacher by 30%. All four
runs take `--seeds-per-dataset 256` from each of the four v16 corpora, which the
mirror corpus reaches only after its extraction completes -- so the seed-0 run
carries the resuming extract and the other three are ordered behind it.
Report per-agent holdout NLL and unit accuracy, per-agent built-in score rates,
and the pairwise policy disagreement matrix on a fixed state batch (the
`_agreement` measure in `scripts/probe_policy_drift.py`).
*Acceptance:* all four within noise of each other on holdout NLL and on
`starter` score rate, and initial pairwise disagreement recorded as the
calibration point for the Stage 3 gate.

*Measured.* Sixteen BC members -- eight seeds of the plain clone and eight of the
NextLat-KL clone (`NEXTLAT_AUX_PLAN.md`), same corpora, same schedule -- scored on
one shared batch of 2,048 active unit decisions from a 4-game rollout
(`artifacts/probes/population-floor.json`):

| block | mean off-diagonal | range |
|---|---|---|
| within the plain arm (n=8) | 0.01133 | 0.01056 - 0.01209 |
| within the KL arm (n=8) | 0.01078 | 0.00972 - 0.01161 |
| across the two arms | 0.01097 | 0.00930 - 0.01230 |
| first four plain members, an actual N=4 population | 0.01128 | -- |

**The floor is 0.011.** So a BC-seeded population starts at about 1.1% pairwise
decision disagreement, 81x below the ~0.894 that four independent random
initializations would give, and the Stage 3 gate's absolute floor is
`0.25 * 0.011 = 0.0028`. That is a safe setting: identical policies read exactly
0.0, and the spread of the measure itself (+-10% within an arm) sits far above
the margin, so the gate cannot misfire on sampling noise.

**But the measure is blind to what decides games, and the gate must not be read
as a diversity check.** Cross-arm disagreement (0.01097) is indistinguishable
from within-arm (0.01133, 0.01078), while the two arms play completely
differently: 3 of 8 KL members beat `public-v27` and 0 of 8 plain members do,
and the winners hold a positive bank margin where every plain member is deeply
negative. Two policies differing on 1.1% of decisions can be a total-loss agent
and a v27-beating agent, because a 719-step compounding economy integrates a 1%
decision difference into an entirely different trajectory.

Two consequences the later stages have to carry:

1. `MINIMUM_POPULATION_DISAGREEMENT` is a *mirror-play detector only*. It catches
   members that have literally become one policy. It cannot certify that a
   population above the floor is behaviorally diverse, and no reading of it should
   be used to argue that.
2. Cloning one teacher does not produce diverse members in any sense this measure
   can see, so diversity has to be engineered and then verified by play, not by
   disagreement. The cheapest lever available is already measured: seeding half
   the population with the KL auxiliary and half without yields members that are
   behaviorally distinct at identical disagreement, at +10.6% per BC epoch.

**Stage 2 - population wave.** `collect_mixed_play_rust` gains a population mode:
per-row agent assignment from the balanced pairing schedule, every row stored,
`RolloutBatch.agents` populated, one ensemble forward over N lanes, no frozen or
built-in lanes. *Acceptance:* a wave of G = 156 returns 312 trajectories with an
exactly balanced 12-pairing histogram and exactly balanced seat counts per agent;
reward antisymmetry holds pairwise (`reward[i] == -reward[j]` up to the shaping
term for every game); CPU test coverage for the schedule and the partition.

**Stage 3 - N-agent training loop.** N actors, critics, and optimizer pairs;
per-agent updates and per-agent gates (`_gate_update_metrics`, including
`MINIMUM_POLICY_ENTROPY`, applied per agent); checkpoint payload carries a list
of agents with a bumped `CHECKPOINT_FORMAT_VERSION`; telemetry emits per-agent
categories plus a `population/` category carrying the N x N head-to-head score
rate matrix and the pairwise disagreement. A new gate,
`MINIMUM_POPULATION_DISAGREEMENT`, stops a run whose members have converged into
each other — the failure mode that silently restores mirror play, where every
other metric reads healthy and every reward is 0.5. Its threshold is set from the
Stage 1 measurement, not invented. *Acceptance:* one event file per run, no
accordion past nine charts, `_LAYOUT_EPOCH` bumped; the disagreement gate fires
on a synthetic population of four identical copies.

How to read that threshold, measured rather than assumed. The disagreement
measure now lives in `src/kaggriculture/policy.py` as `greedy_disagreement`,
`population_disagreement` and `mean_off_diagonal` — the share of *active*
decisions whose masked argmax differs, so an illegal action holding the largest
raw logit never counts. On real artifacts it reads: `bc5` against
`ppo-bc5mix/checkpoint-000040` exactly **0.000**, which is the correctness check
(iteration 40 precedes the actor unfreezing, so the weights are identical), and
`bc5` against `league-actor-00000049` **0.008** at a bank of 14,048 against
85,051. Eight decisions in a thousand cost 83% of the money, and that
checkpoint's unit entropy is 0.227 on the states it visits against 0.018 on the
states BC demonstrated — a 13x gap, which is covariate shift measured directly
rather than inferred.

So the floor is a tripwire and not a target: four clones of one corpus differing
in about a percent of decisions will diverge far past that within a few
iterations of learning, and a gate at a quarter of the initial value should
essentially never bind. If it does bind, the population has genuinely collapsed
into one policy.

**Stage 4 - retire the frozen-league scheduler.** Delete from the training path:
`select_league_mix` and its call site, the PFSP lane contest and built-in lane
admission, snapshot writing, and the historical/active pool selection. Keep the
native built-ins (evaluation and parity audits), the checkpoint writer, and the
stacked-ensemble machinery. *Acceptance:* no training-path caller of the retired
functions remains, the suite passes without them, and no second scheduler is left
behind.

**Refused, not deferred. Validation ran and the replacement lost.** This stage
was sequenced before the launch to stop a second scheduler running silently
beside the first. That hazard never existed: `_validate_population`
(`scripts/train_ppo.py:350`) already *refuses* `--league-games`,
`--league-builtin-lanes` and `--league-builtin-opponents` whenever `N > 1`, so a
population wave cannot reach the frozen-league scheduler at all. The paths are
mutually exclusive by construction and the acceptance criterion "no second
scheduler is left behind" was met by refusal before any code was written.

The deletion was then held for Stage 5 on the grounds that the replacement had
never trained. It has now trained, once, and it is the worse recipe: the
frozen-league path produced every artifact in this repository -- the v27 clone,
BC5, the v16 clones -- while the population path took four artifacts that each
beat `public-v27` and returned four that lose to it 0-of-16, one of them holding
7 money against `starter`. `population == 1` returns early from the same
validator, so the league scheduler *is* the N=1 path.

Deleting the only demonstrated recipe in favour of a measured-worse one is not a
cleanup, so this stage is withdrawn rather than rescheduled. It becomes live
again only if some later change makes a population wave beat the N=1 league on
external score rate, and the fact that both collapse for the same
scale-invariance reason says the reward anchor comes first either way.

**Stage 5 - measured launch. Ran, and the tripwire fired.** `runs/pop4-klwin`,
N=4 at 156 games (13 x 12 ordered pairings), no league and no built-in lanes,
members the four v16-KL clones each individually beating `public-v27`, critic
warmup 40, external evaluation of all four every 10 iterations. Cancelled at
iteration 131 after 90 actor-active iterations decided it.

| it | self-play money | mean self-play score | per-agent range | disagreement | entropy | vs `public-v27` | vs `public-v16` | vs `starter` |
|---|---|---|---|---|---|---|---|---|
| 40 (frozen) | 82,946 | 0.5000 | 0.378-0.603 | 0.063 | 0.000 | **0.875** | 0.750 | 1.000 |
| 60 | 26,218 | 0.5000 | 0.308-0.718 | 0.387 | 0.150 | 0.062 | 0.000 | 1.000 |
| 80 | 29,760 | 0.5000 | 0.308-0.910 | 0.402 | 0.170 | 0.000 | 0.000 | 1.000 |
| 100 | 31,551 | 0.5000 | 0.231-0.821 | 0.421 | 0.183 | 0.000 | 0.000 | 0.750 |
| 120 | 41,333 | 0.5000 | 0.179-0.782 | 0.398 | 0.185 | 0.000 | 0.000 | 0.750 |

The acceptance criterion was inverted on every line. Against `public-v27` the
population went 0.875 to 0.000 in twenty actor-active iterations, and its own
bank fell 83,432 to 15,094 while the *opponent's rose* 79,356 to 132,468: the
members stop competing for the shared economy and let a fixed opponent take it.
Agent 2 at iteration 120 finished with 7 money against `starter`, having beaten
it 1.000 at the warm start.

**Withdrawn: "the decisive number is that the mean self-play score is exactly
0.5000".** It is an accounting identity, not a measurement. `_relative_score` is
antisymmetric and the schedule plays every ordered pairing equally often, so
`score(i,j) + score(j,i) = 1.000000` for every pair and the mean over all twelve
is 0.5 whatever the members are -- verified on iteration 135's journal. One god
and three rocks also average 0.5. The column is in the table above for shape
only; it carries no information and nothing should be concluded from it.

What the *ordered* matrix shows, which the mean cannot, is that the population
was not balanced at all. At iteration 135: `0>1` 0.654, `0>2` 0.692, `0>3`
0.808, `1>2` 0.769, `1>3` 0.846, `3>2` 0.692 -- a strict order 0 > 1 > 3 > 2
with strong edges, so there was a real internal gradient and it was being
followed. Cycling appeared transiently at iteration 81 (0 > 1 > 2 > 0, all three
dominated by 3) and resolved by 135. So the run does not support a
cycling explanation either.

Every internal instrument read healthy or improving while it happened: entropy
alive and rising 0.000 to 0.185, pairwise disagreement up 6.6x, critic explained
variance 0.87, 28 of 28 minibatches applied, `max_approx_kl` 0.005-0.029 against
a 0.10 bound. `MINIMUM_POPULATION_DISAGREEMENT` reads *healthiest* exactly when
the policies are worst, because diverging to exploit each other is diversity.
Only the external anchor detected anything, which is the argument for it.

**Stage 6 - ablations.** PFSP versus uniform pairing; population size; and, if
adopted separately, the reward anchor.

## Risks

**Reward saturation is real, but it is NOT established as the cause of either
collapse. Both runs share one confound: no non-saturating opponent was ever in
the training wave.** Every reward term is scale-invariant --  `_relative_score`
and the dense shaping potential built on it (`terminal_pair_potential`) -- and
absolute competence is therefore a null direction of the objective. That much is
a property of the code and is not in question. What was claimed here, and is now
withdrawn, is that two collapses had established it as the operative cause.

The confound, found by asking why a standard league does not behave this way:

| run | built-in opponents in the training wave |
|---|---|
| `runs/ppo-bc5mix` | `pass,random,starter` |
| `runs/pop4-klwin` | none (`_validate_population` refuses lanes at N>1) |
| `production.py:35` default | `pass,random,starter,scripted-v27` |

`scripted-v27` -- the in-engine port of the strongest known opponent, native in
the batched wave (`core.rs:1226`), already the production default -- was omitted
from the one run that could have used it and structurally forbidden in the
other. And the three that were used all *saturate*, measured in
`artifacts/probes/pop4-anchor.json`:

| policy | vs `starter` | vs `scripted-v27` |
|---|---|---|
| `warm0` (it 40) | 1.000, bank 140,592 | **0.4375**, 66,821 / 72,778 |
| `fin0` (it 130) | 1.000, bank 54,471 | 0.000, 40,279 / 111,832 |
| `fin1` (it 130) | 1.000, bank 54,842 | 0.000, 43,235 / 113,859 |

`starter` reads 1.000 across a 2.6x spread in absolute bank. A constant score
contributes a zero advantage and therefore no gradient, so the "12.5% built-in
rows" in `ppo-bc5mix` were not an anchor -- they were wasted rows. Effectively
both runs were unanchored self-play. `scripted-v27` is the opposite: 0.4375
against the warm start and 0.000 against both collapsed finals, which is a large
corrective gradient that was available and never sampled.

This also repairs the mechanism story. Against a strong opponent the collapsed
policy does not deny -- `scripted-v27`'s own bank *rises* 72,778 to 113,859, the
same direction `public-v27`'s did in external evaluation. Denial only pays
against a market-dependent opponent (`bc5`, itself weaker than `public-v27`); a
strong opponent is denial's beneficiary. So denial is an exploit of a specific
opponent class, not a dominant strategy, and an exploit is exactly what a league
with a permanent non-saturating anchor is supposed to punish.

That is the AlphaStar recipe this plan claimed to be following and was not: a
supervised anchor kept in the league permanently, main agents spending a large
share of games against the whole past league, and PFSP up-weighting opponents
you are losing to. Beating a fixed strong opponent requires absolute capability,
which pins the null direction without touching the reward at all.

**Therefore the reward change is no longer the next step.** Anchoring touches
`rust/kagg_env/src/core.rs` and the meaning of every trained artifact; it must
not be spent on a hypothesis confounded by a missing flag. The decisive and far
cheaper experiment is `ppo-bc5mix`'s own configuration with `scripted-v27` added
to the built-in lanes -- one variable, against a known collapse. Only if a
policy still walks off its bank with a non-saturating anchor in the wave does the
reward become the suspect.

**Market denial is still positively rewarded. Measured on this run's own
members.** `artifacts/probes/pop4-denial.json`, 16 games per ordered cell, two
seat orders per edge, warm start member 0 (`warm0`, iteration 40) against final
members 0 and 1 (`fin0`, `fin1`, iteration 130):

| policy | bank in mirror play |
|---|---|
| `warm0` | 61,772 / 66,360 |
| `fin0` | 48,416 / 52,439 |
| `fin1` | 50,087 / 48,816 |

Ninety actor-active iterations made every member **poorer against an equal**.
`fin1` then beats `warm0` 0.844 without ever out-earning it: facing `warm0` it
holds 47,933 -- below its own mirror bank -- while cutting `warm0` from 61,772
to 28,767, and from the other seat `warm0` collapses to a 8,633 median. The
winner of that edge is the poorer policy, which is the pathology stated exactly:
a scale-invariant reward pays for the opponent's loss at any price in your own
bank, and the market is one shared `market_inventory` whose price is a function
of it (`rust/kagg_env/src/core.rs:269,2869`), so dumping product is a permanent
transfer away from both players and toward whoever needs it less.

**Cycling. Present but transient, and it explains nothing.** The risk was that
non-transitive drift among four learners is unchecked and the aggregate score
rate cannot see it. Both halves need correcting.

The cross-generation matrix is a three-cycle at 32 games per edge -- `warm0`
beats `fin0` 0.719, `fin0` beats `fin1` 0.812, `fin1` beats `warm0` 0.844 -- so
non-transitivity is real between generations. But the *live* population's own
ordered matrix was cyclic only at iteration 81 and strictly ordered by 135
(0 > 1 > 3 > 2, above), so the run ended with a clean internal ranking.

And this was claimed to explain the 0.5000 aggregate. It does not: that number
is an antisymmetry identity and needs no explanation. What survives is narrow
and still useful -- aggregate and per-agent score rate both read 0.5 in a cycle
and in a converged tie, so neither can detect non-transitivity, while the
ordered pairwise matrix can. The run already journals it as
`population_score_rate_i_vs_j` and nothing consumed it. Read it; do not treat
the mean as data.

**Convergence of the population.** Four agents on the same architecture, corpus
and objective can drift together until the league is mirror play under another
name. This is why `MINIMUM_POPULATION_DISAGREEMENT` exists and why Stage 1
measures the initial value rather than guessing a floor. Now measured at 0.011,
which also bounds what the gate is for: it detects members that have become one
policy, and nothing finer. Members differing on 1.1% of decisions were measured
winning 3-of-8 against `public-v27` versus 0-of-8, so this gate passing is not
evidence that a population is still exploring different strategies. That claim
needs play.

Stage 5 settled the direction of that gap, and it is worse than uninformative.
Disagreement rose 6.6x, 0.063 to 0.42, over exactly the iterations in which
every member fell from 0.875 to 0.000 against `public-v27`. The gate reads
*healthiest* when the policies are worst, because diverging in order to exploit
each other is diversity by this measure. It is a mirror-play detector and must
never be read as an exploration signal or a run-health signal.

**Replay parity under vmap.** The behaviour policy moves from a plain compiled
module to `vmap` over `functional_call`. If that shifts logprobs beyond the
shipped ceilings, the ratio in the surrogate is measuring the wrong thing. Gated
in Stage 0 with an existing instrument.

## Decisions taken

1. **`N = 4` total**, each agent facing the other three.
2. **Cost parity, at 156 games.** Measured, not chosen on taste
   (`artifacts/probes/population-cost.json`, `scripts/probe_population_cost.py`,
   on `runs/bc5-mixed-conv/bc-actor.pt`, 3 repeats, median):

   | wave | games | forwards/step | rollout s | trajectories per learner | host arena |
   |---|---|---|---|---|---|
   | today's mixed | 112 self-play + 96 league | 2 | 4.07 | 320 | 3.19 GiB |
   | population G156 | 156 | 1 | **3.08** | 78 | 3.11 GiB |
   | population G636 | 636 | 1 | — | 312 | **12.70 GiB, over budget** |

   Two results decide it. Data parity is not a compute call at all: at 636 games
   the host arena wants 12.7 GiB against a 12.0 GiB budget on a 32 GiB machine,
   so it is infeasible before any throughput question arises. And cost parity is
   not merely affordable, it is **24% cheaper per wave than what ships today**,
   because every seat is a live learner and the second forward -- the frozen
   ensemble's -- disappears entirely (`forwards_per_step` 2 -> 1).

   The honest cost is stated rather than buried: each member sees 78 trajectories
   per iteration where the single learner sees 320, so per-member sample
   efficiency drops roughly fourfold and a population run needs proportionally
   more iterations for the same per-member experience. What it buys is that all
   78 come from live opponents of equal competence rather than from frozen
   snapshots of its own past. Both waves pass every admissibility gate
   (`worst_first_minibatch_kl` 2.5e-05, `worst_update_replay_max_kl` 4.4e-04,
   11x under the shipped replay ceiling), so vmap replay parity is not the
   constraint either.
3. **The reward gets no absolute anchor.** Only winning counts, so the objective
   stays purely relative and antisymmetric: `_relative_score` and the shaping
   potential built on it are unchanged, and `rust/kagg_env/src/core.rs` is not
   touched. "My economy grew but by less than my opponent's" is the signal the
   scheme is meant to carry, and a relative reward carries exactly that.

   The consequence is explicit rather than hidden: mutual mediocrity remains a
   fixed point, market denial remains positively rewarded whenever the opponent
   is beatable, and no internal number can see either. Stage 5's tripwire --
   absolute bank must not fall while a relative score rate rises -- is therefore
   not a safety net but **the primary detector**, and the external evaluator is
   the only absolute measurement in the system.

   What the population does fix is the defect the measurements actually
   condemned: 93% of the last run's experience came from the learner's own
   current or frozen weaker weights, so beating them measured nothing. Every
   opponent is now a live peer of equal competence that improves as the learner
   improves, and no pairing is zero by construction.
