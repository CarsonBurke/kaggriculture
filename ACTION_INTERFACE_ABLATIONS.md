# Action-interface ablations, 2026-09-22 to 2026-09-30

This is the single plan and queue ledger for action-decoding ablations.
`RUNS.md` records measured outcomes and broader training history. Background
measurements below remain useful, but the current gates and job IDs in section
4 supersede the original dated schedule.
Final submission deadline: 2026-09-30 23:59 UTC. The plan freezes the
submission candidate at 2026-09-29 12:00 UTC so a full day is left for building
and validating the bundle.

**2026-09-23 update.** Stage-0b falsified the proposed A1/A3 fixed market
ordering. On Kaggle 1.32.7 against public-v27, unchanged v16 won 128/128 paired
games and banked $92,496 on average; fixed, impact-sorted and HIRE-last
rewrites each won 0/128 and banked $8,477, $8,578 and $2,200. The original
claim in section 1.5(4) that sells-first is safe is wrong: order changes hiring
and purchasing budgets, and repeated/interleaved kinds are sometimes essential.
Do not use the Stage-0 gate rule below to choose among those three variants.
Any A1 market relabel needs exact full-engine replay equality. Current A3 is a
parity-tested opt-in prototype whose corpus builder requires an explicit lossy
relabel flag; it is not a training candidate. The exact-order causal decoder,
A2 quantity alias, and per-head diagnostic panels have completed their first
attempts. See section 4 for current jobs.
The first causal-v4 PPO attempt was stopped after only four actor-update waves:
normal rollout cost 15–17 seconds versus 2–3 seconds for flat, then changing
frozen-opponent lane counts forced 175- and 204-second recompiles. Its custom
CUDA choice/distribution ops also lacked vmap batching rules. This is a
throughput failure, not a learning result; a fresh causal run needs a matched
production-shape speed gate before training.
If the exact-order decoder remains too slow after vmap and graph-mode fixes,
the next design candidate is a **market-only causal** decoder: calculate all
unit logits in one neural pass, select/apply units sequentially through the
exact ledger, then run cached attention only for the 20 ordered market
kind/quantity choices. This keeps the teacher's slot order and repeated kinds
while removing 16 expensive attention steps. Its CUDA parity gate passed and
its short forward probe is faster; two-epoch BC and bounded PPO are queued.
**2026-09-24 priority update.** The A2 BC showed an argmax gain, but its PPO
job failed on the first update from GPU OOM. Matched A2 retry jobs were
cancelled at the user's request so the market-only causal variant can be
developed next. The percentage quantity head in section 2.2b is being built
as the next independent quantity ablation.

## 1. What the evidence says before any arm runs

### 1.1 The interface as it is (verified in code)

- Units: 68 `UnitAction` values per unit, up to 16 units ([actions.py:35](src/kaggriculture/actions.py#L35)).
  36 are `PICKUP_<ITEM>_<N>`; only fills the shed can cover are legal
  ([actions.py:285](src/kaggriculture/actions.py#L285)). Moves are primitive and legal anywhere in
  bounds, locked tiles included, which matches the engine
  ([core.rs:2154](rust/kagg_env/src/core.rs#L2154)).
- Market: 22 `MarketKind` values, 10 slots, STOP always legal and terminal
  ([actions.py:114](src/kaggriculture/actions.py#L114),
  [core.rs:1849](rust/kagg_env/src/core.rs#L1849)). Slot logits come from ten learned slot
  queries in one forward pass ([entity.py:449](src/kaggriculture/entity.py#L449)); a slot sees earlier
  slots only through the ledger mask.
- Quantity: 100 absolute bins ([constants.py:175](src/kaggriculture/constants.py#L175)). The logit is
  a full `bias[kind, q]` plus a rank-32 kind-gated term
  ([model.py:625](src/kaggriculture/model.py#L625)), and it is evaluated **inside the Rust sampler**
  ([core.rs:2815](rust/kagg_env/src/core.rs#L2815) `score_quantities`), not on the GPU. Unit and kind
  logits come from the GPU (Gumbel utilities, [rollout.py:900](src/kaggriculture/rollout.py#L900)) and
  Rust only masks and selects them ([core.rs:1919](rust/kagg_env/src/core.rs#L1919)).
- Engine: units, then market. Market slots of both players advance in lockstep; within a slot both
  players are quoted the same price for each unit round, then commit in seat order
  ([core.rs:2361](rust/kagg_env/src/core.rs#L2361)). Slot index therefore matters against the
  opponent only across slots: my wheat sell in slot 0 is filled at better prices than the opponent's
  wheat sell in slot 1. v27 exploits this by ranking its sells by price impact
  ([core.rs:1492](rust/kagg_env/src/core.rs#L1492)).
- Days: hands are deleted at the end of every day and the farmer respawns at (4,4)
  ([core.rs:2557](rust/kagg_env/src/core.rs#L2557)); hands must be re-hired daily at Fibonacci cost.
  Units never block each other ([core.rs:2154](rust/kagg_env/src/core.rs#L2154)), so shortest
  paths between two tiles differ only in intermediate positions. The one place an intermediate
  position matters is `spawn_hand`, which puts a new hire on the least-occupied shed-access tile
  given current unit positions ([core.rs:2962](rust/kagg_env/src/core.rs#L2962), called from
  `hire` after the unit phase at [core.rs:2489](rust/kagg_env/src/core.rs#L2489)). Replaying 16
  episodes with canonical paths changed 13 of 2,773 intermediate positions and 0 of 848 hire
  spawns, so paths are outcome-equivalent on this corpus but not in principle.

### 1.2 The production corpus is one trajectory

The production clone trains on `data/bc-v16-current-{mirror,starter,pass,random}-64`, 512
episode-seats of 719 steps. (`bc-v16-current-v27-64` has a manifest and no episodes;
[RUNS.md:3041](RUNS.md#L3041) confirms it was never used.) Each file stores the factored targets,
masks, **and the raw observations and engine actions** (`raw_json_zlib`), so any interface can be
re-projected on CPU without re-running the teacher.

| Measurement over the 512 episode-seats | Value |
| --- | --- |
| Active unit decisions equal to that (step, unit) cell's mode across episodes | 99.89% |
| Distinct whole-turn unit vectors per step, mean over steps | 1.54 |
| Market-kind decisions equal to the cell's mode | 99.96% (99.79% over active slots only) |

Consequences. Holdout BC NLL on this corpus measures memorization of a clock-indexed script, so it
cannot rank parameterizations by itself; every BC arm must be judged closed loop. BC relabeling for
any interface is exact and cheap.

### 1.3 What the teacher's actions look like

| Statistic (512 episode-seats) | Value |
| --- | --- |
| Active decisions per turn | 11.56 (9.27 unit, 1.83 market kind, 0.46 quantity) |
| Moves / PASS / pickups, share of unit decisions | 42.8% / 15.9% / 1.98% |
| Move runs (maximal same-day runs of one unit) | 736,248, mean length 1.99 |
| Runs of length 1 / at most 2 / at least 5 | 57.6% / 75.0% / 9.0% |
| Runs that are shortest paths (share of runs / of move steps) | 99.0% / 97.2% |
| Two-axis runs that are L-shaped / that move along x first | 97.3% / 98.2% |
| Runs ending at a day boundary / in PASS / at the shed | 1.8% / 4.7% / 5.1% |
| Turns with no market order / with at least two | 62.7% / 13.3% |
| Turns using all 10 slots | 2,560 (5 per episode) |
| Turns with a repeated kind | HIRE 23,552; BUY_PRODUCT_WHEAT 3,584; SELL_WHEAT, SELL_FERTILIZER, BUY_SEED_WHEAT 512 each |
| Turns with both sells and buys: sells first / buys first / interleaved | 65.3% / 34.6% / 6 turns |
| Multi-sell turns in ascending kind order | 40.9% of 12,952 |
| Sells at the legal maximum / buys below it | 64.1% / 94.5% |
| Legal quantity bins per decision, mean: buys / sells | 53 / 11.7 |
| Distinct quantities used / largest | 30 / 46; 1, 2, 3 are 29.0%, 14.4%, 12.6% |
| BUY_LAND | steps 160 and 240 only, every episode |

Hindsight target relabeling, measured on 32 episode-seats (the corpus is deterministic, so more adds
nothing): the action that ends a move run was already legal at its tile when the run started in
99.79% of runs (the rest are plants whose seed is bought mid-route and one cow pickup, one each per
episode), and some non-PASS verb was legal there in 99.93%. Two units share a live target (the
tile their current action executes on) in 30.9% of turns and 12.7% of unit-turns; about half of
those unit-turns are at the shed and the rest are genuine co-location, such as FEED and CARE on
the same animal.

### 1.4 Where the shared clone's sampled departures sit

Departure mass is 1 - p(teacher choice) under teacher forcing, summed per game: the expected
number of T=1 departures on teacher states. It is computed from the shared BC initializer
(`artifacts/probes/credit-valuation-20260918/bc-actor.pt`, two epochs) on 8 episode-seats.

| Component | Decisions per game | Departure mass per game |
| --- | ---: | ---: |
| Unit, teacher moved | 2,855 | 0.64 |
| Unit, teacher did a verb or PASS | 3,806 | 2.18 (1.03 of it where the teacher PASSed) |
| Market kind, teacher STOPped | 714 | 1.26 |
| Market kind, teacher placed an order | 598 | 2.46 (HIRE only 0.07) |
| Quantity, teacher at the legal max / below it | 332 | 0.62 / 0.35 |
| Whole step, joint over all components | 719 | 7.44 |

Market heads carry 63% of the mass. The largest single flow is 1.88 per game of probability moved
onto STOP when the teacher placed an order, concentrated on SELL_MILK and SELL_WOOL, where the teacher
sells and the clone is unsure whether to sell this turn. That is timing uncertainty, not encoding
redundancy (milk and wool are co-sold once per episode). No interface removes timing uncertainty.
What an interface controls is **what a departure costs**. Under STOP-termination, one mistimed STOP
also drops every later order that turn; under a per-kind set, a mistimed milk sell costs only the
milk. A primitive-move departure strands a unit off an open-loop script with no recovery data; a
target-pointer departure costs a step or two, and the compiler paths the unit back from wherever
it is.

### 1.5 Premises in the brief that the data contradicts

1. **"The pointer cuts per-route decisions about 8x."** Routes average 1.99 steps and 58% are a
   single step; the dominant pattern is step, water, step, water. A Markov pointer is still one
   decision per unit per step. The case for the pointer is robustness and a better readout (1.4,
   2.4), not fewer decisions. Moves carry only 0.64 of the 7.44 departure mass on teacher states.
2. **"Possibly a same-turn claimed-tile mask."** The teacher co-targets tiles in 12.7% of
   unit-turns, half of them away from the shed. An exclusivity mask would make the teacher
   unrepresentable. Drop it.
3. **"Cheap reparameterizations keep the Rust sampler unchanged."** That holds for unit and kind
   logits, which are GPU-computed. It is false for quantity: any quantity reparameterization changes
   `score_quantities` and the head-shape checks in
   [python.rs:474](rust/kagg_env/src/python.rs#L474) and
   [python.rs:728](rust/kagg_env/src/python.rs#L728). The change is small (section 2.2).
4. **"Sells first so proceeds fund buys" as the teacher's convention.** v16 buys before selling in
   35% of mixed turns. The original plan inferred that sells-first would be a
   safe canonical order from own-ledger feasibility. The official paired panel
   disproved that inference: market slot order changes actual fills, subsequent
   budgets and open-loop teacher outcomes. A1 must require exact full-engine
   replay equality before publishing any market relabels.
5. **"A production PPO iteration is about 6 s."** That is the lejepa family (6.4 s median over
   64 iterations, `runs/lejepa-full-20260922`). The production entity actor runs at 12.1 s median
   (`runs/structural-gae-20260918/component-control`, about 2.45 s rollout and 9.4 s update).
6. **"Market is interleaved unit by unit."** Within one slot both players are quoted the same price
   each unit round, so a slot's execution is symmetric. What orders across slots is slot index, and
   only for price-moving kinds (the nine SELLs and the two BUY_PRODUCTs).

## 2. Arms

Structural policy changes are opt-in model-config fields or versioned action
interfaces so old checkpoints retain their behavior. Matched arms use the same
rules, corpus, opponent schedule and evaluation seeds. All BC runs use two
epochs; GPU training has a 30-minute hard cap and each speed benchmark a
two-minute hard cap.

| Arm | Change | State | Next gate |
| --- | --- | --- | --- |
| F0 | Flat schema-v4 control | Built; matched BC/PPO queued | Compare with market-causal |
| C0 | Full 36-step causal decoder | Built; first PPO stopped for throughput | Retain as diagnostic, no retry yet |
| C1 | Parallel unit logits, 20 ordered causal market choices | Built; CUDA parity and eager speed probe passed | Two-epoch BC, matched PPO and panels queued |
| A2 | Explicit ALL quantity alias | Built; BC promising; PPO OOM | Paused after user reprioritization |
| A2b | Fraction-structured integer quantity policy | Implementation in progress | Exact Rust/Python likelihood and replay parity |
| A1 | Engine-equivalent corpus relabeling | Market rewrites rejected | Exact full-game replay before any path-only corpus |
| A3 | Per-kind market set in a fixed order | Prototype built; official paired panel rejected it | No training of this compiler |
| A4 | Target-tile pointer | Planned; unit-sampling diagnostic completed | Reassess after C1/A2b; do not queue yet |

### 2.1 A1: canonicalized corpus (control for everything after it)

**Design.** Re-project the 512 raw episodes with three outcome-preserving canonicalizations and
change nothing else.
(a) Paths: every maximal shortest move run is relabeled to the x-first L path to the same
endpoint in the same number of steps. Relabeling is done per state: at each step of the run the
target is the x-first step from the unit's actual position toward the run's endpoint, not a
spliced canonical action sequence. This is outcome-equivalent on the corpus but not in principle
(`spawn_hand`, 1.1), so the round trip replays each canonical episode in the engine and requires
final-state equality with the original. Non-shortest runs (2.8% of move steps) are split greedily
into maximal shortest sub-runs, each canonicalized, which keeps every waypoint at its original
step.
(b) Market: merge same-kind orders into one summed order (HIRE stays repeated, because interface 1
has no count), then order as sells (SELL_WHEAT..SELL_FERTILIZER), HIRE, BUY_LAND, seeds, animals,
products, or by the Stage-0b ordering rule if that rule wins.
(c) Nothing else changes.

**Why.** It removes path and permutation ambiguity at the data level with no model or sampler
change, and it is the control that separates "canonical data" from "set-valued head" in A3.

**Touchpoints.** New `canonicalize_demonstration` beside `project_demonstration`
([demonstrations.py:368](src/kaggriculture/demonstrations.py#L368)); the round-trip gate
([demonstrations.py:551](src/kaggriculture/demonstrations.py#L551)) changes from byte equality to
own-ledger equivalence for merged or reordered markets, plus the engine replay and final-state
check above. New dataset directories carry
`format_version` 2 in their manifest; [train_bc.py:104](scripts/train_bc.py#L104) accepts it. The
encoded cache is keyed by manifest hash and needs no change.

**Risk.** Merging changes order execution relative to the opponent. Stage 0b measures that at the
teacher level before any model sees it.

### 2.2 A2: an ALL encoding inside the existing quantity categorical

**Design.** Add one "ALL" row to the quantity head: `all_value[rank]` and `all_bias[kind]`, scored
exactly like a bin. The legal maximum m is the last true bin of the prefix mask, which Rust already
guarantees is a prefix ([core.rs:3903](rust/kagg_env/src/core.rs#L3903)). Masks cap at 100 bins
([core.rs:2752](rust/kagg_env/src/core.rs#L2752)), so ALL means min(legal maximum, 100); for buys
that is often 100. Inactive rows have an all-false mask, so m is taken as max(mask.sum() - 1, 0)
and the merge is masked out with the row. The effective logit of
bin m becomes `logaddexp(score(m), score(ALL))`; every other bin is unchanged. This is the marginal
likelihood over the two encodings that compile to the same engine order, so BC keeps plain NLL on
the unchanged targets. It applies to every quantified kind; the learned bias decides where ALL
matters (64% of sells, 5.5% of buys).

**Why this form.** An additive "bonus on the max bin" has the same expressiveness, but its
parameter means something different at every m. The mixture gives ALL its own state-dependent score.
Stored factors, masks and trajectory arrays are untouched, and m is recoverable from the stored
quantity mask at replay. An ordinal (discretized-logistic or cumulative-link) head is not proposed.
The corpus's non-max quantities are specific small integers (1-3 are 56% of all quantities),
which the per-kind bias already captures, and the only max-relative structure is ALL.

**Touchpoints.** `factored_quantity_logits` gains the mask argument and the merge
([model.py:625](src/kaggriculture/model.py#L625)). Every `quantity_logits` caller changes with it:
[ppo.py:1680](src/kaggriculture/ppo.py#L1680), [ppo.py:1724](src/kaggriculture/ppo.py#L1724),
[ppo.py:2008](src/kaggriculture/ppo.py#L2008), [train_bc.py:881](scripts/train_bc.py#L881),
[train_bc.py:931](scripts/train_bc.py#L931), [train_bc.py:1257](scripts/train_bc.py#L1257),
[train_bc.py:1290](scripts/train_bc.py#L1290),
[causal_actor.py:336](src/kaggriculture/causal_actor.py#L336),
[causal_actor.py:532](src/kaggriculture/causal_actor.py#L532), the benchmark and probe scripts,
[latent_dynamics.py:302](src/kaggriculture/latent_dynamics.py#L302),
[actor_dynamics.py:180](src/kaggriculture/actor_dynamics.py#L180). Python sampling
([policy.py:616](src/kaggriculture/policy.py#L616)) and `PreparedQuantityHeads`
([policy.py:98](src/kaggriculture/policy.py#L98)). In Rust, `QuantityHead` gets the ALL row
([core.rs:423](rust/kagg_env/src/core.rs#L423)), and `score_quantities` merges at m
([core.rs:2815](rust/kagg_env/src/core.rs#L2815)). The head-shape checks at
[python.rs:474](rust/kagg_env/src/python.rs#L474) and
[python.rs:728](rust/kagg_env/src/python.rs#L728) become `[heads, 101, rank]` and
`[heads, 22, 101]`. A wave never mixes interfaces: `FrozenActorPool` builds every league slot from
the learner's own model config and loads strictly ([league.py:307](src/kaggriculture/league.py#L307)),
and snapshots are validated against it. `_quantity_heads`
([rollout.py:401](src/kaggriculture/rollout.py#L401)) should still reject a stack with mixed
`action_interface` values explicitly. Padding interface-1 heads with `all_bias = -inf` is not an
option: the select path rejects non-finite quantity tensors
([python.rs:808](rust/kagg_env/src/python.rs#L808)).
Parity: extend the Rust-vs-Python quantity likelihood tests and the replay-parity audit.

**Risk.** Low. It is also the fallback deliverable if A3 slips.

### 2.2b A2b: percentage-structured market quantities

**Hypothesis.** Quantities are ordered, and the legal maximum changes with the
state and preceding orders. Sharing a policy over the fraction of that maximum
may generalize across legal caps better than 100 unrelated categorical rows.
This remains an ablation, not a replacement for A2: 56% of corpus quantities
are exactly 1–3 and 64% of sell quantities are the legal maximum, so one plain
unimodal Beta can fit the important modes poorly. The Beta policy in
`../cleanrl/cleanrl/ppo_continuous_action.py` uses
`alpha,beta = 1 + softplus(head)`, which does not create endpoint spikes.

**Proposed first arm.** Keep the categorical kind/STOP decision, and for a
selected quantified kind define its positive legal amount `q` in `1..m`.
Use a small mixture with explicit atoms at 1, 2, 3 and `m`, merging duplicates
when `m < 4`; model the remaining amounts by a continuous CDF over `u` in
`(0,1)`. For `q = ceil(m*u)`, its integer probability is the CDF difference
`F(q/m) - F((q-1)/m)`. A discretized logistic CDF is the first candidate;
a discretized Beta or Beta-binomial can be compared if it offers a measurable
fit gain without expensive or unstable CDF evaluation. The current local
PyTorch 2.13 build has no `torch.special.betainc`, so a Beta CDF would need
additional differentiable implementation work. Normalize the mixture
after masking illegal or duplicated atoms. When `m = 1`, `P(q=1) = 1` and
the quantity entropy is zero. Zero amount stays in the kind/STOP decision;
adding 0% to this head would duplicate a no-order decision.

**Likelihood contract.** BC uses the exact probability of the teacher's
executed integer amount. Native sampling, recorded rollout log-probabilities,
PPO replay, entropy and KL use that same integer distribution and the same
prefix-specific `m`. Evaluating a Beta density at the center of a rounded
integer is incorrect. CleanRL's raw Beta log-density is valid for PPO only if
the exact sampled latent percentage is stored and replayed; that would not
provide a direct likelihood for the existing integer BC demonstrations and
would waste exploration on percentages that execute identically. Require
Python/Rust parity at legal caps 1, 2, 3, 16 and 100, with boundary and
duplicate-atom cases, before training.

**Readout.** Compare A2b with both the original categorical control and A2 ALL
using two-epoch BC, held-out quantity NLL by kind and legal cap, the same
paired argmax/sampled native panels, trio sales, and matched PPO capped at
30 minutes. Benchmark each inference path within two minutes. Promote only on
closed-loop outcome and sale volume, not BC NLL alone.

**Other amount-like actions.** Unit pickup actions encode bounded quantities
(wheat up to 16, fertilizer up to 8, animals up to 4). Audit their frequency
and current per-head departure cost before adapting this head. Movement,
planting, HIRE and BUY_LAND are discrete choices in this engine; continuous
observation features do not imply continuous action distributions.

### 2.3 A3: the market as a per-kind canonical set

**Design.**
- One decision per kind, sampled in a fixed ledger order: the nine SELLs, then HIRE, BUY_LAND, the
  five seeds, the three animals and the two products. Sells go first because proceeds and freed shed
  room only widen later masks.
- Each quantified kind chooses q in {0, 1..m} with the ALL mixture from A2. HIRE chooses a count in
  {0..h}, where h is the largest count affordable under cumulative Fibonacci cost, the 15-hand cap
  and the slot budget. BUY_LAND chooses {0, 1}; the engine allows more per turn, but the teacher
  never does.
- A kind whose only legal value is 0 is inactive (not a decision, no gradient), as quantities are
  today.
- The ledger carries a slot budget, so the compiled queue never exceeds 10 slots (the official
  `maxMarketOrdersPerTurn`, [mechanics/constants.md:15](mechanics/constants.md)).
- The compiler emits slots as the active sells first, then HIRE times n, LAND, seeds, animals and
  products. Sells go in fixed kind order, or in price-impact order (v27's rule, a deterministic
  function of state and the chosen set) if Stage 0b shows the ordering is worth money.
- STOP, permutations and duplicate kinds no longer exist as choices. A mistimed decision on one kind
  cannot truncate the others.

Heads: the ten slot queries ([entity.py:359](src/kaggriculture/entity.py#L359)) become 21 kind
queries, so every kind has its own decision state. Each kind's rank-32 context feeds the existing
kind-gated quantity machinery, extended to a 101-row value table (row 0 means "none") plus the ALL
row. HIRE and LAND reuse the same machinery over their small supports. The `market_kind` linear head
is deleted in interface 2.

**Why this and not the alternatives.** A causal (autoregressive) kind decoder
([causal_actor.py](src/kaggriculture/causal_actor.py)) also fixes conditioning. However, it keeps
the order redundancy, keeps STOP-truncation, and costs a sequential decode. The prior review reached
the same conclusion ([REVIEW_RL_20260920.md](REVIEW_RL_20260920.md) section 5.4).
A Plackett-Luce priority over active price-moving kinds is the exact-likelihood way to learn slot
order. It is added only if Stage 0b shows that neither fixed order nor impact order comes within
noise of the teacher's own order.

**Departure-count risk.** A3 raises market decisions from 2.3 to roughly 10-15 active per turn:
seeds are affordable almost always, and sells are active whenever stock exists. Each must hold
p(0) near 1. The existing kind head already has to reject every non-teacher kind at every slot, so
the difficulty is similar, but it must be measured, not assumed: departure mass per game on
canonical teacher states (section 3) is a gate.

**Touchpoints.** Python reference: `MarketKind`/`market_order`/`compile_action` and the ledger
helpers ([actions.py:114](src/kaggriculture/actions.py#L114),
[actions.py:379](src/kaggriculture/actions.py#L379)-[481](src/kaggriculture/actions.py#L481),
[actions.py:667](src/kaggriculture/actions.py#L667)-[722](src/kaggriculture/actions.py#L722)), and
the market loop of `act_batch` ([policy.py:579](src/kaggriculture/policy.py#L579)-[686](src/kaggriculture/policy.py#L686)),
which is also the submission path via [inference.py:423](src/kaggriculture/inference.py#L423). Rust:
the market halves of `factor_masks`, `sample_factors` and `select_factors`
([core.rs:1683](rust/kagg_env/src/core.rs#L1683), [core.rs:1813](rust/kagg_env/src/core.rs#L1813),
[core.rs:1995](rust/kagg_env/src/core.rs#L1995)), `fill_market_*_mask` and
`apply_policy_market_order` ([core.rs:2685](rust/kagg_env/src/core.rs#L2685)-[2813](rust/kagg_env/src/core.rs#L2813)),
and a set-to-slots compiler producing `CompactAction` (`process_market` is untouched). The bindings
change market array shapes and drop the GPU kind utilities
([python.rs:426](rust/kagg_env/src/python.rs#L426), [python.rs:682](rust/kagg_env/src/python.rs#L682),
[python.rs:1451](rust/kagg_env/src/python.rs#L1451)) and
[rollout.py:900](src/kaggriculture/rollout.py#L900), [rollout.py:1079](src/kaggriculture/rollout.py#L1079).
Also affected: `ActionFactors` ([policy.py:64](src/kaggriculture/policy.py#L64)), PPO component
bookkeeping and replay parity ([ppo.py:2336](src/kaggriculture/ppo.py#L2336)), the BC projection
and loss ([demonstrations.py:368](src/kaggriculture/demonstrations.py#L368),
[train_bc.py:868](scripts/train_bc.py#L868)), and lejepa, which splits its decision states by `MAX_MARKET_ORDERS`
([lejepa_model.py:229](src/kaggriculture/lejepa_model.py#L229),
[lejepa_model.py:360](src/kaggriculture/lejepa_model.py#L360)). Only A2 is inherited by lejepa
for free, because A3's kind queries live in the entity trunk. `CausalActor` and `StrategicActor`
share `_initialize_heads` but decode markets themselves
([causal_actor.py:283](src/kaggriculture/causal_actor.py#L283)), so they must reject interface 2
at construction. The action encoders used by the world-model families
(`StructuredActionEncoder` in [lejepa.py:382](src/kaggriculture/lejepa.py#L382),
[structured_dynamics.py:88](src/kaggriculture/structured_dynamics.py#L88),
[economic_forecasting.py:219](src/kaggriculture/economic_forecasting.py#L219),
[latent_dynamics.py:73](src/kaggriculture/latent_dynamics.py#L73),
[actor_dynamics.py:74](src/kaggriculture/actor_dynamics.py#L74)) embed slot-shaped market actions
and need a set-shaped variant or a compile-to-slots adapter. About 30 Python test files and 40 Rust
tests pin the current interface; interface 2 adds tests beside them rather than editing them.

**Risks.** (1) The departure count above. (2) Every interface-2 league snapshot must be
interface 2. Production league lanes are the run's own snapshots plus Rust built-ins, so this holds.
(3) Three days is the estimate with parity tests; it is the critical path of the plan.
(4) PPO scale: entropy is normalized by the active component count
([rollout.py:1079](src/kaggriculture/rollout.py#L1079)) and the policy loss is a per-component
mean, so going from about 2.3 to 10-15 market components per state changes the effective step size
and entropy weight of every head family under an unchanged recipe. Stage 3 reports per-family
approximate KL and entropy for every arm; if A3's market KL per wave is outside a factor of two of
A0-12's, its entropy coefficient and clip range are recalibrated on one short run before the seeded
comparison.

### 2.4 A4: target-tile pointer (gated on Stage 0a)

**Design.** Each unit makes one flat categorical choice over 64 local verbs (the current actions
minus the four moves, PASS included) plus 100 target tiles. Choosing a verb acts on the current
tile exactly as today. Choosing a target T (never the current tile) compiles to the first step of
the x-first shortest path, the teacher's own convention in 98.2% of two-axis runs, re-decided every
step (Markov). Engine-level move probability is the sum over targets: P(EAST) is the total mass on
targets with larger x. About 0.07% of teacher targets are not representable (no legal verb at the
endpoint at route start); those steps are relabeled with the next waypoint as target.
- Target mask: T is legal when some non-PASS verb would be legal for this unit at T under the
  current same-turn ledger. Because DIG is legal on every plant or weed and BUILD on every empty
  unlocked tile, this mask is essentially "unlocked tiles plus shed access minus quiet animal
  tiles". It computes in O(100) per unit from per-tile flags, not by 100 calls to the full action
  mask. Locked tiles are never targets but remain transit, as the engine allows.
- There is no claimed-tile mask (1.5.2).
- Pointer logits: the unit decision state as query against the own-farm tile tokens from the trunk
  memory ([entity.py:463](src/kaggriculture/entity.py#L463)), with the existing axial RoPE applied to
  both sides so the score sees relative displacement.
- BC relabel: every step of a move run gets its endpoint as the target, the action at arrival is the
  verb, and non-shortest runs split into waypoints as in A1.

**Why it could matter even though moves carry little departure mass.** A pointer readout is
position-equivariant over tile tokens, which primitive moves are not. A mistimed or wrong step
leaves the plan intact instead of stranding the unit off an open-loop trace. Under PPO, a sampled
exploration target is a coherent errand instead of a random walk. None of these shows up in
teacher-forced metrics, which is why A4 is gated on the closed-loop cost measured in Stage 0a.

**Risks.** Dithering between targets at T=1. It shows up as a rise in steps per completed errand
in sampled play; the fix is conditioning on the unit's previous target, which is a unit-feature
addition and is deferred. The unit mask grows from [16, 68] to [16, 164] bools: 604 MB instead of
250 MB per 230k-state wave, which is acceptable, or bit-packed if memory binds.

**Touchpoints.** `UnitAction` space and `unit_action_mask`/`compile_action`
([actions.py:35](src/kaggriculture/actions.py#L35), [actions.py:255](src/kaggriculture/actions.py#L255),
[actions.py:685](src/kaggriculture/actions.py#L685)); the unit loop of `act_batch`
([policy.py:519](src/kaggriculture/policy.py#L519)-[570](src/kaggriculture/policy.py#L570)). Rust
`UnitLedger::action_valid` gains the target class, and the unit halves of `factor_masks`,
`sample_factors` and `select_factors` split the policy factor from the engine action they compile
to ([core.rs:493](rust/kagg_env/src/core.rs#L493), [core.rs:1654](rust/kagg_env/src/core.rs#L1654),
[core.rs:1740](rust/kagg_env/src/core.rs#L1740), [core.rs:1919](rust/kagg_env/src/core.rs#L1919)).
`UNIT_ACTIONS`-sized arrays change in the bindings, in `_gpu_policy_statistics`, and in the unit
head ([entity.py:598](src/kaggriculture/entity.py#L598)) and `decode_belief`, which must now receive
tile states ([entity.py:641](src/kaggriculture/entity.py#L641)). The pickup initialization bias in
[model.py:575](src/kaggriculture/model.py#L575) is retained for the verb block.

### 2.5 Not proposed, and why

- **Pickup factorization.** Pickups are 1.98% of unit decisions and carry about 0.14 of the 7.44
  departure mass. The only cheap form, `logit = item + count` inside the existing 68-way head, is
  pure model code (unit logits are GPU-side) and can ride along with A4 if it is built. It is not
  worth an arm.
- **Ordinal quantity head.** Covered in 2.2.
- **Claimed-tile mask.** Contradicted by the teacher (1.5.2).

## 3. Evaluation protocol

All panels are the fixed native development panel (256 maps, seeds from `DEVELOPMENT_SEED_START`,
both seats) ([evaluate_architecture_campaign.py](scripts/evaluate_architecture_campaign.py)). Every
number is reported as score and paired bank (learner minus the same-map baseline), with a
map-paired bootstrap 95% interval.

**BC metrics, comparable across interfaces.** Evaluate every arm on the same holdout episodes
(the highest seeds, which [train_bc.py:221](scripts/train_bc.py#L221) holds out), with equivalence
defined at the level of what the step does, not how it is encoded.
- Step-level engine-action NLL: -log of the probability that the arm's policy produces a step
  equivalent to the teacher's. Units are equivalent when the compiled engine action is equal
  (summing over targets that share a first step). The market is equivalent when the own-ledger net
  effect is equal: the multiset of (kind, total quantity), with ALL and m identified. For slot-order
  arms that probability is a sum over the orderings and splits that reach the teacher's net effect;
  it is computed exactly by enumerating them, which is cheap because the teacher's turns have at
  most a handful of distinct kinds. Without this, A0-12 would be charged for its learned buys-first
  and split orders and the A1+ arms would pass gate (ii) by construction.
- Expected departures per game: the sum over steps of 1 - p(teacher's engine action). This is the
  quantity section 1.4 decomposes, and the one tied to the sampled tail.
- Per-head top-1, only as a regression check.

**Closed loop.** Argmax and T=1 sampled play against starter and scripted-v27 on the 256-map panel.
The public-v16 panel runs in the official engine (64 seeds, both seats, CPU workers via
[evaluate_checkpoint.py](scripts/evaluate_checkpoint.py)) for final candidates only.

**PPO.** Screen each arm from its own two-epoch clone for at most 27 training
minutes (30-minute hard MLQ limit). Keep optimizer, minibatch, critic warmup,
league schedule and evaluation seeds matched within each comparison. Read out
both argmax and sampled native play, paired v27 bank, per-head entropy and
carrot/tomato/egg sale counts. Multiple training seeds and official-engine
games are required before a final submission choice; one-seed development
panels screen arms, not certify them.

## 4. Current build and queue plan (2026-09-24)

This section is the authoritative action-ablation plan. `RUNS.md` contains
observed results, not competing future schedules. All jobs use normal MLQ
priority and exclusive GPU admission. A job's queue wait does not count toward
its cap. Two epochs is the BC default; no benchmark exceeds two minutes and
no training run exceeds 30 minutes.

### 4.1 C1: market-only causal, first priority

The source is frozen at
`artifacts/source-snapshots/fe1078ae45b47d2fb831eba4ede8156009a1422a0abd21968159b9b73a8de1cb`.
The opt-in `parallel_unit_decode` flag retains exact sequential unit ledger
updates, computes the unit neural logits together, and makes the 20 market
kind/quantity choices causally. CUDA native-mask/replay parity **9526** passed.
At 192 midgame rows, the two-minute eager probes found **70 ms** per C1 single
forward (**9527**) versus **129–150 ms** for C0 (**9528**). C1's two- and
four-lane eager ensembles took 129–132 and 141–143 ms with zero vmap fallback
warnings. C0's eager ensembles still fail efficient attention's 36-column mask
stride; the C1 market cache pads to 24 columns. These forward numbers do not
establish whole-rollout or PPO speed.

| Arm | Two-epoch BC | PPO | Argmax / sampled panels | Trio sales |
| --- | ---: | ---: | ---: | ---: |
| Matched flat v4 F0 | **9533** | **9534** | **9537 / 9538** | **9541** |
| Market-only causal C1 | **9535** | **9536** | **9539 / 9540** | **9542** |

The PPO jobs depend on their own BC jobs; panels and sales depend on PPO
success. Commands and frozen source are in
`artifacts/probes/market-causal-v4-20260924/campaign.json`. Both PPO runs use
component ratios, minibatch 4096, default Inductor update compilation without
CUDA graph capture, four critic-warmup iterations, and the same league setting:
one active frozen-opponent lane, zero historical lanes and the native builtin
opponents. That stable lane count avoids the previous 1→2→4 ensemble
recompilations; it changes the training distribution from the growing league,
so all conclusions are within this matched comparison. Inspect the first
actor-update wave and stop an arm if replay parity fails or actor updates make
no progress. Do not infer play strength from forward latency.

### 4.2 A2b: percentage-structured quantities, second priority

Implement the exact integer policy in section 2.2b as opt-in interface 4. The
market kind/STOP factor stays categorical; an active quantity takes one of
`1..m`, where `m` is the current legal maximum after all preceding orders.
Python BC, PPO replay and the Rust native selector must agree on the same
per-integer probability and entropy. The first candidate uses a discretized
logistic CDF plus explicit small-quantity and ALL atoms, merging duplicate
atoms when `m` is small. The existing categorical interface 1 and ALL
interface 2 are controls. A raw Beta log-density on a rounded integer is not
an acceptable replay likelihood. Current PyTorch has no `special.betainc`,
so Beta CDF integration is a later candidate only if the logistic arm misses a
specific pattern.

Queue in this order after implementation: (1) focused CPU/Rust parity and
boundary/gradient tests; (2) exclusive CUDA replay contract, two-minute cap;
(3) matched two-epoch F0/A2b BC; (4) matched, at-most-30-minute PPO only if
BC and native sampling are legal; (5) paired argmax/sampled panels and trio
sales. Every command and dependency goes into one campaign manifest beside the
outputs. Compare by legal cap, kind and executed quantity as well as outcome;
NLL alone cannot promote an arm.

### 4.3 Other action candidates and stop gates

- **A2 ALL:** Its BC argmax panel was promising, but PPO failed at the first
  update with CUDA OOM. Retry jobs 9518–9524 were canceled at the user's
  request. Keep it as a control for A2b; do not restart that PPO campaign while
  C1 and A2b run.
- **A1 path relabel:** The market-order relabel failed exact engine-equivalence
  checks and is excluded. Build a path-only corpus only after full native
  trajectory replay proves identical next states; then compare to a fresh
  two-epoch control. Do not queue the rejected market rewrite.
- **A3 fixed-order per-kind set:** The official paired panel decisively
  rejected it (0/128 wins against v27 versus 128/128 for the unchanged
  teacher). Its parity-tested prototype remains archived; no BC/PPO job for
  that compiler.
- **A4 target-tile pointer:** Completed per-head panels **9483–9487** show
  units-only sampling improves v27 score to 26.95%, versus 17.19% for all
  sampled and 0% for argmax. Kinds-only remains at 0%, so unit exploration is
  useful while sampled market kinds are the clearer harm. The old numerical
  A4 gate is satisfied, but its premise that unit departures are damaging is
  not. Reassess A4 after C1/A2b; require an exact factor likelihood and native
  first-step mapping before queueing BC. Movement remains a discrete grid
  decision, even if a pointer selects a distant target.
- **P1 pickup amount:** Current pickup counts are discrete with maxima 16/8/4.
  Audit their opportunity and regret contribution first. Adapt the A2b integer
  mass head only if this is material; do not apply a raw continuous PPO density
  to a rounded pickup count.

### 4.4 Promotion and resource rule

A clear winner in paired native outcome and relevant trade volume becomes the
initializer/control for later action runs immediately; record the source and
checkpoint digest. A one-seed argmax gain or held-out BC NLL gain alone is a
screening result. Keep both decoding modes and the carrot/tomato/egg units in
every market-related panel. Training jobs stop at 30 minutes; benchmarks stop
at two minutes. Do not launch a new long run merely to obtain a benchmark
number that would take longer to compile than to measure.

## 5. Biggest uncertainties

1. **Whether any interface fixes what PPO is losing.** The review attributes the PPO erosion
   mostly to objective, credit and drift. Interface arms can shrink the sampled tail and still not
   make PPO improve over BC. Stage 3 is designed to show that honestly, not to rescue it.
2. **C1's actual training throughput.** The eager forward is faster and CUDA
   parity passed, but full rollout, Inductor compilation and PPO updates have
   not yet been measured for the new decoder.
3. **Fraction-policy credit.** A2b shares amount structure across legal caps,
   but the teacher favors exact 1–3 and legal maximum. Explicit atoms and
   exact integer likelihood may still add complexity without improving play.
4. **Timing uncertainty is untouched by quantity parameterization.** The largest departure flows (STOP versus
   SELL_MILK/SELL_WOOL, PASS versus act) are when-to-act questions. If A0-12 already removes most
   of them, interface gains will read small at BC.

## Appendix: reproducing the measurements

Scratch scripts, run with `CUDA_VISIBLE_DEVICES= .venv/bin/python -P <script> <corpus dirs>` under
the 16 GB memory scope. They are not part of the repository.

- `corpus_action_stats.py`: sections 1.3 and 1.5.
- `corpus_diversity.py`: section 1.2, duplicate kinds, and path axis order.
- `pointer_relabel_stats.py`: hindsight relabel legality, shared targets, PASS waits.
- `departure_mass.py <bc-actor.pt> <corpus dirs>`: section 1.4.
