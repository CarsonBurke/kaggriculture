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

**Stage 5 - measured launch.** Cost check against Stage 0's table, then the run,
with external evaluation of all N agents. Instrument cycling without adding a
frozen lane: in evaluation only, score each agent against its own weights from K
iterations earlier, which detects non-transitive drift without giving history any
gradient. *Acceptance:* per-agent external score rate against `public-v27` and
the built-ins rising over iterations; per-agent absolute bank not falling while a
relative score rate rises — the exact pathology of the last run.

**Stage 6 - ablations.** PFSP versus uniform pairing; population size; and, if
adopted separately, the reward anchor.

## Risks

**The population does not fix reward saturation, and cannot.** Every reward term
is scale-invariant: `_relative_score` and the dense shaping potential built on it
(`terminal_pair_potential`). A population of four equals that all collapse to the
3,000 starting bank scores 0.5 each, exactly as mirror self-play did. What the
scheme removes is the *guarantee* of a zero signal from identical weights; what
it does not remove is the fixed point at mutual mediocrity. Mitigation is an
absolute bank anchor in the reward, which touches `rust/kagg_env/src/core.rs`
and the meaning of every trained artifact, and is therefore a separate decision.
Until then, Stage 5's acceptance criterion — absolute bank must not fall while
relative score rises — is the tripwire.

**Market denial is still positively rewarded.** Measured above: destroying the
shared market beats out-farming when the opponent is beatable. Four co-learners
do not change the incentive, and the first member to discover denial wins its
pairings. The Stage 5 tripwire catches it; only an absolute anchor removes it.

**Cycling.** With no past players, non-transitive rock-paper-scissors drift among
four learners is unchecked, and the aggregate score rate cannot see it — a
three-cycle keeps every member near 0.5 forever. Instrumented in Stage 5 by
scoring each agent against its own earlier weights in evaluation only.

**Convergence of the population.** Four agents on the same architecture, corpus
and objective can drift together until the league is mirror play under another
name. This is why `MINIMUM_POPULATION_DISAGREEMENT` exists and why Stage 1
measures the initial value rather than guessing a floor.

**Replay parity under vmap.** The behaviour policy moves from a plain compiled
module to `vmap` over `functional_call`. If that shifts logprobs beyond the
shipped ceilings, the ratio in the surrogate is measuring the wrong thing. Gated
in Stage 0 with an existing instrument.

## Decisions taken

1. **`N = 4` total**, each agent facing the other three.
2. **Cost parity or data parity is still open** and is the one question left for a
   measurement: Stage 0(c) produces the ms/step table and the choice is a compute
   call, not a design call.
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
