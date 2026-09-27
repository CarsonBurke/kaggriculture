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
its short forward probe was faster, but its two-epoch BC unit fit was much
worse than flat. The user closed the causal arm on 2026-09-24; all remaining
causal jobs were canceled before start.
**2026-09-24 quantity priority update.** The memory-safe A2 ALL PPO retry
succeeded; percentage and narrow-percentage heads failed their sampled-game
gates, and the transplanted percentage head failed its early-buy gate. The
promoted initializer is now the A8 local-affordance PPO actor with categorical
ALL quantities. The latest factorwise decode and pickup audits in section 4.3
support retaining that head while using greedy quantities when evaluating an
otherwise sampled policy. PPO rollout and deterministic submission remain as
trained; a new quantity head needs stronger evidence of an amount-specific
failure before another training run.

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
| F0 | Flat schema-v4 control | BC completed; causal-campaign PPO canceled | Historical reference only |
| C0 | Full 36-step causal decoder | Stopped for throughput | Closed by user |
| C1 | Parallel unit logits, 20 ordered causal market choices | BC unit fit weak; all later jobs canceled | Closed by user |
| A2 | Explicit ALL quantity alias | PPO **9588** completed and promoted within A2 | Follow-up unit/kind split **9626** and matched BC sales **9627** |
| A2b | Fraction-structured integer quantity policy | Two-epoch BC **9589** succeeded; scale-floor NLL ceiling found | PPO **9590** running; paired panels queued |
| A2c | Narrow-scale fraction head | CPU/native parity passed, immutable source frozen | CUDA gate **9617**, BC **9618**, panels **9619/9620** queued |
| A1 | Engine-equivalent corpus relabeling | Market rewrites rejected | Exact full-game replay before any path-only corpus |
| A3 | Per-kind market set in a fixed order | Prototype built; official paired panel rejected it | No training of this compiler |
| A4 | Target-tile pointer | Planned; unit-sampling diagnostic completed | Reassess after quantity results; do not queue yet |

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
quantity mask at replay. A separate discretized-logistic quantity arm is now
specified in section 2.2b.
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
- **Raw continuous quantity density.** Section 2.2b explains why the game
  requires an exact executed-integer likelihood even when a latent fraction is sampled.
- **Claimed-tile mask.** Contradicted by the teacher (1.5.2).

## 3. Evaluation protocol

Primary panels use the fixed native development panel (256 seeds from
`DEVELOPMENT_SEED_START`, seat determined by seed parity). Two-minute
preliminary gates may use its first 128 seeds, as A2c does; require a larger
confirmation before promotion
([evaluate_architecture_campaign.py](scripts/evaluate_architecture_campaign.py)). Every
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
  most a handful of distinct kinds. Without this, a slot-order control would be charged for its learned buys-first
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
observed results, not competing future schedules. Training uses normal MLQ
priority and exclusive GPU admission; short defect probes **9626/9628/9636**
were raised one priority level to resolve the current action failure. A job's
queue wait does not count toward
its cap. Two epochs is the BC default; no benchmark exceeds two minutes and
no training run exceeds 30 minutes.

### 4.1 C1: market-only causal, closed

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
| Matched flat v4 F0 | **9543** | **9553** | **9554 / 9555** | **9556** |
| Market-only causal C1 | **9548** | **9557** | **9558 / 9559** | **9560** |

C1 BC **9548** finished with holdout NLL **0.0482** and unit accuracy
**0.977**, versus F0 **0.0025** and **1.000**. The parallel unit path bypassed
the decoder's farm/economy attention, a likely capacity cause. C1 PPO
**9557** and dependents **9558–9560** were canceled before start. A repaired
unit decoder passed CPU replay/gradient tests, but the user closed the causal
arm; its queued CUDA contract **9577** was canceled before start, and dependent
probe/BC **9578/9579** skipped. The uncommitted repair was removed from the
working tree. F0 PPO **9553** was also canceled during the causal-campaign
shutdown; its BC **9543** remains a measured reference. Earlier superseded
chains and commands are retained in
`artifacts/probes/market-causal-v4-20260924/campaign.json`.

### 4.2 A2 ALL versus A2b percentage: active priority

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

Compare A2 and A2b with the same LeJEPA actor family, four-corpus BC recipe,
PPO update settings, evaluation seeds and current rules. A2's existing
two-epoch BC checkpoint is the control and the best initial argmax result.
A2b receives its own two-epoch BC. Both PPO jobs have a 27-minute soft and
30-minute hard limit. Read paired argmax/sampled panels and trio-sales volume;
NLL alone cannot promote an arm.

The opt-in interface-4 implementation is frozen at
`artifacts/source-snapshots/5c1e129eb7697502dfe2965aaac02d2de6453f2ce8d0bdba1b6ad97b980c1ffe`.
It parameterizes a logistic fraction conditioned to [0, 1] and adds 1, 2, 3,
and legal-maximum atoms, merging atoms that execute the same integer amount.
Torch BC/PPO, NumPy inference, and Rust native sampling use the resulting
100 integer logits. Sixteen focused CPU tests, selected PPO/BC replay tests,
Rust build, Ruff and independent review passed; the Rust scale softplus was
made stable at large learned values and tested at raw scale 100. Native parity
has so far covered an opening-state legal maximum; Torch/NumPy cover maxima
1, 2, 3, 16 and 100. The CUDA legal-logit gate **9562** passed in under one
second of test time within its two-minute cap; BC is now eligible to train.

| Quantity job | MLQ ID | Prerequisite | Hard cap |
| --- | ---: | --- | ---: |
| CUDA integer-logit gate, passed | **9562** | None | 2 min |
| A2 ALL memory-safe PPO, succeeded | **9588** | Existing A2 BC **9474** | 30 min |
| A2b LeJEPA two-epoch BC, succeeded | **9589** | Gate success | 30 min |
| A2b LeJEPA PPO, canceled | **9590** | A2b BC success | 30 min |
| A2 argmax / sampled panels, succeeded | **9591 / 9592** | A2 PPO success | 20 min each |
| A2 carrot/tomato/egg sales, succeeded | **9593** | A2 PPO success | 10 min |
| A2b argmax / sampled panels, skipped | **9594 / 9595** | A2b PPO success | 20 min each |
| A2b carrot/tomato/egg sales, skipped | **9596** | A2b PPO success | 10 min |

All jobs use normal priority and exclusive GPU admission. Commands and
dependencies are recorded in
`artifacts/probes/quantity-focus-20260924/campaign.json`. Both PPO arms use
component ratios, minibatch 4096 and default Inductor update compilation
without CUDA graph capture, avoiding A2's earlier first-update OOM. The
previous entity-architecture percentage chain **9567–9571** was
canceled/skipped before start because it would confound the quantity comparison
with an actor-family change.

**A2 low-money tail diagnostic.** The matched 256-game sampled BC panels show
an A2 v27 median of $38,115 versus $17,638 for the categorical LeJEPA control,
but 42 versus 6 games finish below $1,000. A2's first ten critic-only PPO
waves have median money p10 about $444; waves 38–47 rise to about $7,392.
This establishes a fragile sampled BC start, not a permanently weak PPO tail.
Head-intervention probes **9606** (A2) and **9607** (categorical) completed
128 paired v27 games each with argmax, quantity-only sampling, no-quantity
sampling, and all-head sampling. A2's below-$1,000 counts are **0, 2, 24,
19** respectively; its money p10 is **$62,553, $53,437, $79, $68**. Thus
quantity sampling alone produces a strong, mostly robust actor, while
sampling units and kinds creates the low-money tail even with argmax
quantities. The categorical control has **1, 0, 2, 5** below-$1,000 games
under the same four modes. Large legal-maximum buys are associated with A2
failures in both no-quantity and all-sampled modes; this is not evidence that
the ALL quantity atom causes those failures. A2's max-buy hypothesis is
rejected as the primary explanation. In no-quantity sampling, the 24
low-money games average 105 buy and 87 sell orders, versus 137 buy and 133
sell orders in the remaining games. The all-sampled low-money games average
101 buy and 79 sell orders, versus 134 and 128 otherwise. This reduced
activity is consistent with disrupted market timing or production, but does
not isolate STOP from unit moves. Each probe used a two-minute cap and
normal-priority exclusive admission; per-game and market-order traces are in
`artifacts/probes/quantity-tail-20260924/`. A2 unit-only, kind-only,
unit+quantity and kind+quantity split **9626** used 128 matched games and a
two-minute cap to locate which other head drives the tail.
That split completed: unit-only, kind-only, unit+quantity, and kind+quantity
sampling produced **8, 7, 8, 11** below-$1,000 games, with v27 score rates
**0.508, 0.727, 0.461, 0.641**. Unit sampling hurts score more than kind
sampling in this A2 BC actor, while the two heads together produce 24
low-money games. The failure is an interaction, not a market-only or
unit-only defect. Keep A4 and A6 conditional on more focused diagnosis.
Two-minute diagnostic **9636** repeats the split with per-game unit PASS,
MOVE, PICKUP, PLANT, WATER and HARVEST counts and active-unit fractions, plus
nonSTOP orders per turn and SELL/HIRE fractions, to
locate which activity collapses before choosing a structural decoder.
Probe **9636** completed. Unit-only sampling averages PASS on **19.6%** of
active unit decisions, HARVEST on **4.23%**, 0.725 nonSTOP market orders per
turn, and a **10.5%** SELL fraction. Kind-only sampling gives **18.2%**,
**4.95%**, **0.779**, and **11.9%**. These are means of per-game fractions;
the paired unit-minus-kind score difference is **−0.219** (exploratory
bootstrap 95% **[−0.328,−0.109]**). Within the eight unit-only low-money
games, PASS averages **26.9%**, HARVEST **1.43%**, nonSTOP orders **0.542**
per turn, and SELL **6.5%**. The seven kind-only low-money games have the
same qualitative loss of farming and sales. Thus low-money episodes show
production and market activity collapsing together; the aggregate data
do not prove that move direction, PASS, or STOP is the first bad decision.
The simple market STOP calibration arm remains deferred. Next unit work
should audit the first divergent unit decision on paired seeds before a
target-tile pointer or verb factorization receives a BC slot.

A2 PPO **9588** completed successfully at wave 68. Its internal fixed panel
score improved from 0.8008 at initialization to 0.8242 at wave 50, and
rollout money p10 median rose from $444 in waves 1–10 to $11,129 in waves
54–63. Paired endpoint panels **9591/9592** completed: argmax score improved
0.9844 → 0.9902 overall, with a paired 95% score interval crossing zero;
sampled score improved **0.5957 → 0.6504**, paired difference **+0.0547**
with exploratory 95% interval **[+0.0156,+0.0938]**. Sampled v27 money p10
rose **$165 → $8,498**, and games below $1,000 fell **42 → 6** in 256 seeds.
This is a clear within-A2 PPO winner. The promoted immutable A2 reference is
`artifacts/promoted/quantity-all-ppo-20260924.pt`, SHA-256
`c6814f2532b8d5d553cea21d063e3d070bb2fa190035b084533ec64da7258824`.
Use it for later A2-derived runs; keep the independently initialized A2b/A2c
arms matched for their interface comparison. A2 PPO is not yet a clear winner
over the prior categorical PPO, which scored 0.6562 sampled overall on its
historical panel.
The 64-game sampled self-play sales audit **9593** recorded 58 carrot units,
5 tomato units and 0 egg units across 128 A2 PPO trajectories. Thus PPO has
some carrot/tomato trades but has not found the large new-rule trade volume,
and egg remains absent. Matched A2 BC sales audit **9627** completed with
**139 carrot, 2 tomato and 0 egg units** across the same 128 sampled
trajectories. The PPO panel therefore sold fewer carrots and only three more
tomatoes in raw total; it did not discover the desired three-product trade.
These are paired-seed aggregate totals, not a significance claim. Keep the
score promotion separate from the resource-trading objective.

A2b BC **9589** completed two epochs. Holdout quantity NLL is **0.18469**,
versus **0.00435** for matched A2 ALL. Unit NLL is 0.00215 versus 0.00188,
and kind NLL is 0.00412 versus 0.00345. Quantity decisions are only about
4% of active decisions, so the quantity gap explains nearly all of the
overall NLL difference (0.00977 versus 0.00223). The compact fraction head
is substantially harder to fit to the script in two epochs. More specifically,
its logistic scale has a hard minimum of 0.02 of the legal range. Across the
full 512-seat BC corpus, 32,734 of 170,348 quantified decisions (19.2%)
target neither 1, 2, 3 nor the legal maximum. Numerically optimizing
location and scale **separately for each such target**, while respecting the
0.02 floor, gives a quantity NLL floor near **0.1656** averaged over all
quantity decisions. The observed 0.1847 is already close to that optimistic
oracle; longer training is unlikely to close the gap to ALL under this
parameterization. This is a numerical oracle, not a formal analytic bound,
but the code-imposed scale minimum explains most of the measured quantity
fit gap. The calculation is reproducible with
`artifacts/probes/quantity-focus-20260924/quantity_floor_oracle.py` and its
JSON result in that directory. Exact integer parity passed, and the BC
losses alone cannot establish whether sampled gameplay or PPO learning is
worse. Keep the outcome panels as the gate.
The A2b PPO run **9590** has begun. Its fixed initialization panel score is
0.4727, versus A2 ALL's 0.8008 on the matched internal panel. This is an
early closed-loop warning consistent with the broad fraction quantity head;
the first ten PPO waves update only the critic, so treat the later actor-wave
trajectory and paired panels as the outcome comparison.
At wave 1, A2b's sampled market STOP share is 0.712 and sell share 0.051,
versus A2 ALL's 0.580 and 0.106, while mean executed quantity is similar
(5.33 versus 5.14). The immediate weakness therefore includes when and
whether to sell, not just how many units each order specifies. These are
different BC fits sharing a trunk; the comparison does not isolate a direct
quantity-to-kind causal effect.
**A2b is rejected.** The user canceled PPO **9590** after iteration 66; its
dependent panels **9594–9596** were skipped. The first critic-only rollout had
money mean/median/p90 **$3,899/$50/$4,739**, versus A2 ALL's
**$64,153/$62,787/$140,650**. Its STOP/SELL/HIRE order fractions were
**0.712/0.051/0.113**, versus ALL's **0.580/0.106/0.203**; unit PASS/HARVEST
were **0.299/0.019**, versus **0.191/0.041**. This is an economically broken
sampled BC initializer even though unit and kind teacher-forced accuracies
exceed 99.8%. At actor wave 50 the fixed-panel score was 0.500, up from
0.4727 but still well below ALL's 0.8242; its terminal-money recovery is not
enough evidence to promote it. Keep the canceled checkpoint for diagnosis,
not as a run parent. Matched head-intervention probe **9628** used a
two-minute cap to determine whether fraction sampling alone causes the
collapse or requires unit/kind sampling as well.
That probe completed on 128 paired v27 seeds. Percentage BC argmax has
**0.805** score, **$84,106** median money, and 2 below-$1,000 games.
Sampling **only quantity** collapses to **0.016** score, **$37** median, and
**96** below-$1,000 games; all-head sampling gives **0.008**, **$56**, and
**108**. On the same seeds, ALL BC quantity-only sampling has **0.898** score
and just 2 low-money games. Percentage quantity-only sampling raises
legal-maximum buys from 11.1 to 28.6 per game; buys at a legal maximum of at
least five rise from 1.07 to 6.32. Sell orders fall from 150.0 to 47.2 per
game. The quantity-only money is lower than argmax in 124/128 paired games,
with mean paired difference **−$76,212** (exploratory bootstrap 95% interval
**[−$81,523,−$70,701]**). This directly establishes that quantity sampling
is sufficient to cause the percentage arm's economic collapse. The scale
floor explains most of its teacher-forced quantity NLL; A2c's sampled panel
will test whether narrowing it also fixes the observed full-game failure.

**Pre-PPO closed-loop gate for new action interfaces.** Finish the matched
two-epoch BC and a paired sampled full-game panel against the promoted ALL
BC before queueing PPO. Record mean, median, p10 and p90 final money, score,
and fraction of games below $1,000, paired by seed and seat. Reject an arm
whose sampled mean, median, or p10 money is below half of matched ALL BC;
whose below-$1,000 fraction exceeds the control by more than five percentage
points; or whose sampled score drops by more than five percentage points,
unless a specific controlled head intervention shows a recoverable decoding
choice. Also inspect STOP/SELL/HIRE, unit PASS/HARVEST, and legal-maximum
buys in a rollout diagnostic before PPO; the paired outcome-panel schema
does not contain these action fractions.
Teacher-forced NLL and argmax wins cannot override a failed sampled-game gate.
Use this gate on A2c after jobs **9633/9634**; do not automatically queue PPO.

**A2c narrow-scale fraction arm:** version the fraction head so its minimum
scale is 0.001, and keep the four atoms, seven trainable outputs,
initialization, and exact integer mass contract. Existing interface-4
checkpoints retain their 0.02 floor.
The same per-target numerical oracle on the full corpus
drops from NLL 0.1656 at scale floor 0.02 to 0.00033 at floor 0.001,
so this intervention directly removes the identified expressivity ceiling.
The isolated source snapshot is
`artifacts/source-snapshots/9ca98babc31354706ac4584ce6d3a040217ae0d1490382f85c3e289c3c1455ee`.
Fourteen CPU tests covering Torch/NumPy parity and Rust sample/select integer
log-probability parity for both floors passed against its isolated native
build; `cargo check` and release build passed. Independent review found no
static blocker. CUDA parity gate **9617** has a two-minute hard cap; matched
two-epoch LeJEPA BC **9618** depends on gate success and has a 30-minute cap.
CUDA gate **9617** succeeded with six tests in 1.49 seconds of test time.
Paired 128-seed argmax/sampled BC panels **9633/9634** depend on BC success,
each with a two-minute cap; the older 256-seed, five-minute-cap jobs
**9619/9620** were canceled before starting. All use normal-priority
exclusive admission.
Commands and immutable source are in
`artifacts/probes/quantity-narrow-20260924/campaign.json`. Compare this arm
with interface 4 and A2 ALL using the same seeds. Queue PPO only if closed-loop
BC merits it; per-cap NLL is a diagnostic of the corrected floor. Gate job
**9635**, dependent on sampled panel **9634**, writes the explicit per-opponent
money/score decision to `artifacts/probes/quantity-narrow-20260924/gate.json`.

Matched two-epoch A2c BC job **9618** succeeded. Holdout quantity NLL is
**0.05780** and quantity argmax accuracy **98.798%**, versus 0.18469 and
97.602% for A2b percentage BC. ALL BC remains stronger at 0.00435 and
99.953%. The lower scale floor fixes a substantial fit defect but leaves a
material gap. The paired sampled-game gate, not fit alone, decides on PPO.
Initial panel jobs **9633/9634** failed at artifact import before any games:
the evaluator loaded the live package, whose config does not include isolated
interface 5. Corrected jobs **9643/9644** set `PYTHONPATH` to the frozen
source and passed an explicit import smoke check; dependent gate **9645**
replaces the skipped **9635**. All retain two-minute caps.

Corrected paired panels **9643/9644** succeeded. A2c narrow-scale BC is
better than A2b percentage BC under sampled play, but fails the A2 ALL
control decisively. Against scripted v27, argmax score is **0.211** versus
ALL **0.977**, with **27/128** versus **0/128** money-below-$1,000 games.
Sampled score is **0.023** versus ALL **0.336**, with **59/128** versus
**19/128** low-money games. Against starter, sampled score is **0.734** versus
ALL **0.891** and low-money games **24/128** versus **8/128**. No A2c PPO
will run. Two-minute head-intervention probe **9652** is queued to separate
residual quantity sampling from unit/kind drift in this arm.
The explicit paired gate **9645** completed and returned `passes=false` for
both starter and scripted v27; its p10/median/low-money checks all reject
promotion. See `artifacts/probes/quantity-narrow-20260924/gate.json`.
Head intervention **9652** completed on 128 paired v27 seeds. A2c argmax
score/low-money games were **0.211/27**; sampling only quantities yielded
**0.211/24**. Sampling unit and kind while keeping quantity greedy yielded
**0.016/55**, and sampling all heads yielded **0.023/59**. Thus, after the
scale fix, quantity stochasticity is no longer the dominant failure. The
greedy policy is already poor, and unit/kind sampling creates the sampled
tail. A2d's inherited frozen ALL unit/kind/trunk directly tests this drift.

**A2d quantity-head transplant, conditional on A2c:** If the corrected
fraction head still fails the sampled-game gate despite a much lower quantity
NLL, isolate representation drift by starting from the promoted A2 PPO
trunk/unit/kind weights, replacing only its categorical quantity head with
the narrow fraction head, and distilling the parent's exact executed-integer
distribution on the same teacher and on-policy states. Freeze the inherited
heads during the first fit. Compare the transplanted head with the unchanged
parent on paired argmax and sampled full games before any PPO. This is a
controlled head test, not a fresh architecture claim; measure p10 and
maximum buys explicitly. A2c failed that gate, so this arm is active.
The shape change is limited to `market_quantity_value.weight` (101×32 to
7×32) and `market_quantity_bias` (22×101 to 22×7); copy and strictly verify
all other actor tensors. Distill KL over the **masked executed-integer**
distribution, not raw head parameters, on teacher and promoted-policy
on-policy prefixes. Track KL by kind and legal maximum. A dedicated script
must also preserve the LeJEPA objective state if its result will warm-start
PPO; standard BC/PPO loaders do not perform this migration automatically.
The isolated implementation is
`artifacts/probes/quantity-transplant-20260924/distill.py`. CPU checks verified
strict tensor migration, exact KL/gradients on a real encoded row, and loading
all 60 parent objective tensors. Independent review found no launch blocker
after adding bounded fit/evaluation reserves, positive-step checks, and KL
breakdowns by market kind and legal maximum. Distillation **9655** is queued
with a 30-minute hard cap and a 25-minute internal cap; paired 128-seed
argmax/sampled panels **9656/9657** depend on its success, each capped at two
minutes. A partial fit or incomplete holdout evaluation cannot support
promotion even if a checkpoint is saved.
Initial job **9655** failed before training because the rollout stores
`unit_active` among action factors rather than observation states. The
staging code now reads that factor explicitly; a real two-game CPU native
rollout produced 1,438 valid rows and 440 active quantity decisions. The
corrected distillation **9664** is queued with the same 30-minute cap;
dependent paired panels **9665/9666** replace skipped **9656/9657**.
When two unrelated CleanRL jobs left ample GPU memory and utilization low,
**9664** was admitted through `mlq` with `maxParallelRuns=3`; it remains
subject to the same 30-minute cap and is monitored for contention.
Job **9664** succeeded in 61.5 seconds, completing both epochs and all
diagnostics (1,560 optimizer steps). Exact masked integer KL against the A2
parent is **0.220** on 31,939 teacher-holdout quantity decisions and **0.613**
on 8,455 sampled-parent on-policy decisions; on-policy quantity argmax
disagreement is **13.2%**. This is a substantial remaining distribution gap,
not promotion evidence. Paired game panels **9665/9666** were queued as the
outcome gate.
The first paired panel jobs **9665/9666** failed before gameplay because the
distilled artifact omitted mandatory `seed_usage`. Its on-policy collection
also used seed 8,200,000 outside the repository's reserved domains. The
script now validates inherited BC/RL exposure, uses online-RL seeds
20,400,000–20,400,031, and writes the combined provenance. Corrected
two-epoch distillation **9668** is queued with the same 30-minute cap;
paired two-minute panels **9669/9670** depend on it. The earlier artifact
is retained only as a diagnostic and cannot be promoted.
Corrected **9668** completed both epochs with valid reserved seed usage and
complete diagnostics (teacher/on-policy masked quantity KL **0.216/0.626**).
Paired panels **9669/9670** reject A2d: against v27, argmax score falls
**0.984 → 0.000** from promoted ALL PPO to A2d, and sampled score falls
**0.297 → 0.000**. Against starter, argmax mean money falls about
**$153,960 → $13,538**. All nonquantity actor tensors are exact copies, so
the quantity head alone is sufficient to cause this failure. Do not queue
PPO or promote A2d. Paired 128-seed executed-order traces **9671/9672**
are queued to identify which amounts cause the economic collapse.
Those traces completed. On seed 4,501,008, the first greedy order is
`BUY_PRODUCT_WHEAT` with legal maximum 95 for both actors, but ALL PPO buys
**14** and A2d buys **40** (sampled A2d **42**). The next cow/sheep legal
maxima immediately fall from the parent's **6/4** to **4/2**. Across the
128 greedy games, A2d makes **46.1** sell orders/game versus parent
**173.6**, and **75.6** buy orders versus **158.9**. This is an early budget
error and downstream market collapse, not merely sampling noise. Future
quantity heads must pass a paired early-order amount/affordability gate,
especially for high-cap buys, before any PPO queue.

### 4.3 Other action candidates and stop gates

- **A2 ALL history:** Its first PPO failed at the first update with CUDA OOM.
  Retry jobs 9518–9524 were canceled during the earlier causal priority. The
  memory-safe retry **9588** succeeded and its sampled endpoint panel earned
  within-A2 promotion; see section 4.2.
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
  not. Reassess A4 after the quantity comparison. A pointer would replace primitive move
  choices with latent target tiles while retaining the engine's 68 primitive
  actions. BC must marginalize over every target that compiles to the teacher's
  first move; PPO must either store the selected latent target or use the
  corresponding engine-action marginal in both behavior and replay. The target
  mask must permit adjacent locked tiles because the engine permits transit
  through them. Require exact Python/Rust target-to-first-step, sequential
  ledger and likelihood parity over locked transit, two-axis paths and shared
  targets before queueing BC. Movement remains a discrete grid decision, even
  if a pointer selects a distant target.
- **A5 conditional market STOP gate, deferred after A2 split 9626:** The current
  22-way kind choice makes STOP probability depend on the masked set of legal
  nonSTOP kinds. A separate Bernoulli continue decision followed by a masked
  nonSTOP kind choice can calibrate order timing while preserving ordered slots,
  repeated kinds, sequential ledger legality and exact engine-action
  likelihood. Initialize its gate logit from the old masked nonSTOP logsumexp
  minus STOP logit; zero learned residual exactly reproduces the promoted A2
  distribution. Trainable calibration is the ablation. BC targets are the
  existing STOP versus nonSTOP decisions; PPO can store one combined market
  slot log probability. Require Torch/NumPy/Rust sample and selected-logprob
  parity, native teacher-action replay, and a matched sampled BC panel before
  PPO. Algebraically, a scalar gate residual is equivalent to shifting the
  existing STOP logit: this is a calibration/optimization test, not a new
  action distribution, and it does not prevent early STOP truncation. Do not
  spend a BC/PPO slot on it: **9626** found both unit and kind contributions,
  and no evidence yet singles out STOP calibration. A count-first
  decoder is the structural option if early STOP truncation dominates; it
  adds latent replay and infeasible-count questions, so do not mix it into A5.
- **A6 count-first ordered market decoder, deferred:** Predict the
  number of nonSTOP orders (0–10) once, then sample that many ordered kinds
  and quantities under the existing sequential ledger. This preserves
  teacher order and repeated kinds while removing independent early STOP
  draws. The A2 split found a kind contribution but a stronger unit effect
  on score, so isolate premature STOP before implementing this market-only
  change. Store the sampled count as a latent action in PPO and include its
  behavior and replay likelihood; when a sampled earlier order exhausts all
  legal kinds, record both the requested count and forced termination. BC
  uses the teacher's feasible count. Require exact ordered engine-action
  identity, forced-termination cases, likelihood replay, and native parity
  before BC. Do not queue until the market branch is specifically supported.
- **A7 productive-work aliases, gated on corpus ambiguity audit:** Keep every
  existing primitive unit action and add state-dependent `WORK_CROP` and
  `WORK_ANIMAL` latent choices. Under the same sequential unit ledger, each
  alias compiles to a productive local verb such as HARVEST, WATER, FEED,
  CARE or COLLECT in a fixed priority justified against teacher states.
  Merge alias and direct-action probability into the executed primitive's
  exact likelihood for BC/PPO. This gives one learnable intent a stable
  meaning across crop/animal states and may replace random PASS/movement
  departures with useful work; direct primitives preserve coverage. First
  measure how often each alias applies, how often candidate verbs conflict,
  and teacher agreement with the proposed priority. Reject ambiguous
  aliases rather than hard-coding an unmeasured order. If supported, require
  Python/Rust selected-action, replay likelihood and sequential-ledger parity
  before two-epoch BC and sampled full-game gate. Do not combine with a
  movement pointer in the first comparison.

  Initial CPU audit on eight teacher episode-seats (53,330 active unit
  decisions) **rejects a naive crop alias**: among 10,829 teacher crop-work
  choices, HARVEST→WATER→FERTILIZE matches 6,894 (63.7%); with multiple work
  verbs legal it matches only 1,688/5,623 (30.0%). WATER-first improves to
  9,241/10,829 (85.3%) but still misses 1,588, with wheat and strawberry
  preferring opposite actions. A naive animal priority matches 6,940/7,508
  (92.4%) teacher animal-work choices, including 4,726/5,294 (89.3%) in
  ambiguous states. Teacher still PASSES at 2,044 crop and 1,251 animal work
  opportunities and MOVES at 7,614/3,960, so an unconditional work bias is
  unsafe. The best separate fixed priority for each crop agrees on 95.8% of
  teacher crop-work choices, still missing state-dependent cases. Limit
  follow-up to crop/state-aware rules or an animal-only alias
  with a learned work decision, after a broader opportunity audit; no BC job
  for the naive crop alias.
- **A8 action-conditioned local affordances, follow-up to A7 audit:** Keep the
  68 primitive actions and exact native masks. For each legal local verb,
  derive a compact candidate-effect description from the unit, current tile,
  inventory, crop/animal state, and time remaining. A shared scorer compares
  that description with the unit decision state and adds a zero-initialized
  residual to the verb logit. Leave PASS and movement logits untouched in
  this first arm; consider a PASS residual conditioned on available work later.
  The explicit candidate effects give local work a shared representation
  across crops and animals without imposing the rejected fixed verb priority.
  Initialize exactly to the promoted A2 ALL PPO actor. Its full checkpoint
  includes the LeJEPA objective under `structured_dynamics`; transfer that
  to the BC warm-start key `jepa_objective` with exact state verification.
  Audit feature availability and
  stale same-turn information, then require Torch/NumPy feature parity,
  orientation parity, initial logit identity, and exact replayed 68-way
  likelihood before two-epoch BC and sampled full-game gate. This is the
  preferred local-action arm if the narrow quantity work finishes and the
  action failures still matter; no training job is queued yet.
  An isolated rank-16 scorer draft is at
  `artifacts/probes/unit-affordance-20260924/source`. Review removed its
  initial movement residual, added time remaining, and factorized scoring to
  avoid a large action-effect tensor. The conditional PASS residual is
  deferred.
  The isolated PPO loader now permits only the `unit_affordance_scorer`
  false-to-true config change and requires exactly six missing scorer weights;
  both the A2 ALL BC artifact and the promoted A2 PPO checkpoint loaded on
  CPU with exact actor and objective state verification.
  A sixth CPU test compares official Python and native Rust structured tokens
  for both seats with a nonzero scorer, verifies PASS/movement logits remain
  unchanged, and checks native selected-action log probabilities against PPO
  replay to 2e-4 absolute tolerance. All six tests and Ruff pass. A short
  eager BF16 CUDA memory/latency gate **9667** is queued after A2d terminates,
  with a two-minute cap and batch 4096. It compares a full actor policy
  forward/backward against A2 but omits Inductor and the LeJEPA objective;
  a pass is a memory screen, not a full PPO throughput result. No A8 training
  is queued.
  Gate **9667** passed but measured a cold baseline against a warm A8 call.
  Corrected gate **9673** warms both variants at batch 4,096 with 16 active
  units: eager BF16 policy forward/backward is **0.249 s** for A2 and
  **0.258 s** for A8, with virtually identical peak extra allocation
  (**8.573 GB** in both). This removes the dense-scorer memory concern;
  Inductor and the LeJEPA objective still require a separate full-update
  performance gate before a long PPO run.
  A two-minute full PPO-loop smoke **9678** is queued from the promoted A2
  PPO checkpoint with the opt-in scorer, eager rollout/update, two tiny
  iterations, and no critic warmup. It checks loader/objective/replay/update
  integration; its reduced workload is not an outcome comparison or Inductor
  throughput claim. A production-size PPO run remains gated on this result.
  Smoke attempts **9678/9679** failed CLI validation before training; the
  corrected **9681** completed an eager full PPO update and exact replay
  checks, then stopped at its intentionally tiny two-iteration limit before
  the fresh critic met the actor-release gate. Production-shape A8 PPO
  **9682** is queued from the promoted ALL PPO actor and transferred LeJEPA
  objective, with disjoint online-RL seed 20,500,000, a 27-minute internal
  runtime target, and a 30-minute hard cap. Its initial policy is exactly
  the promoted ALL PPO parent; compare only released actor snapshots on
  paired sampled games before any promotion.
  Run **9682** reached actor wave 25 and its fixed panel score moved from
  **0.82031 to 0.83594**, but a later compiled JEPA update failed at
  iteration 45 when `plan.indices` widened to 2,560. This is a compiler
  shape-guard failure, not evidence of action collapse. The immutable
  iteration-35 checkpoint retains the actor, critic, optimizer, RNG, and
  league archive. Recovery job **9686** resumes an explicitly recorded fork
  from that checkpoint with eager updates and the compiled-only architecture
  panel disabled; rollout collection and all action/model parameters stay
  the same. It is exclusive with a 27-minute internal target and 30-minute
  hard cap. A paired external panel is still required for any A8 promotion.
  The fork provenance is preserved in
  `runs/unit-affordance-20260924/ppo-eager-fork/fork_provenance.json`;
  read source metrics through iteration 35 and fork metrics from 36 onward.
  Review of the guard shows `structured_horizon_plan` creates a variable
  number of eligible JEPA pairs per shuffled minibatch; the observed widths
  512, 2,048, and 2,560 are legitimate. The compiled update assumes those
  widths settle and aborts on a new bucket. This is unrelated to A8 action
  logits. A future compiled fix may pad the plan to a fixed bucket while
  preserving its eligible mask and source indices, but needs exact
  loss/gradient checks and a two-minute 4,096-row memory gate before use.
  The row minibatch is already padded to 4,096; the varying dimension is
  eligible adjacent pairs inside the JEPA plan. Full batches usually need
  about 2,048 pairs, but a shuffled run boundary can push the count into
  the 2,560 bucket. A fixed per-batch plan capacity tied to the existing
  `count` loop variable is the narrower proposed fix.
  Two-minute endpoint job **9688** compares the promoted parent, A8 actor
  wave 25, and final A8 actor on 128 new paired seeds per opponent with
  sampled decoding. Two-minute sales job **9689** audits the final actor on
  the parent's 64 self-play diagnostic seeds. Both wait for **9686**.
  Recovery **9686** finished cleanly at iteration 95 in 27.3 minutes. The
  first independent sampled panel **9688** rejects wave 25 (overall score
  difference −0.0273) but favors final A8: score **0.6953 versus 0.6445**
  for promoted ALL, paired difference **+0.0508** with exploratory 95%
  interval **[0.0000,+0.1016]**; mean money difference **+$7,813** with
  interval **[+$2,660,+$13,243]**. Both starter and v27 p10 improved.
  Sales audit **9689** found final A8 **48 carrot, 6 tomato, 0 egg** units
  versus matched parent's **58, 5, 0**; A8 is not a trio-trade solution.
  Replication panel **9690** uses 256 new paired seeds per opponent and a
  two-minute cap before deciding promotion.
  Replication **9690** confirms final A8: overall sampled score **0.6816
  versus 0.6445** (paired **+0.0371**, exploratory 95% interval
  **[+0.0020,+0.0723]**) and mean money **+$6,740** (interval
  **[+$2,903,+$10,711]**). Starter and v27 p10 both rise. Promote the
  immutable final full checkpoint as
  `artifacts/promoted/unit-affordance-ppo-20260924.pt`, SHA-256
  `6d36c49bc7d1e694a6d027b5a67f55abd95e16e53310cc4779a419a275c0a9cd`.
  Its embedded promotion record includes the eager fork provenance.
  Future LeJEPA action ablations now initialize from this A8 checkpoint
  with `unit_affordance_scorer=true`; keep ALL as the paired control.
  The scorer and strict PPO objective warm-start path are ported to the
  canonical source. A8 improves general outcome but is not a carrot/tomato/
  egg-trade solution.
  The promoted artifact retains the isolated training source identity. The
  original paired panels used that source; canonical evaluation now has a
  cross-tree witness, recorded in the A8 quantity follow-up below.
  Warm-starting a fresh canonical run does not rewrite this history. A
  source-locked submission export needs its own bundle validation.
  The paired result promotes the **trained policy**, not a causal claim that
  the scorer beats equal-duration continuation of ALL: A8 received further
  PPO updates and no matched scorer-off continuation ran. Its scorer query
  moved from zero to norm 0.305, showing it was learned, but the separate
  architectural contribution is unresolved. Do not queue a redundant ALL
  continuation merely to relabel this outcome gain as a mechanism win.
- **P1 pickup amount:** Current pickup counts are discrete with maxima 16/8/4.
  Audit their opportunity and regret contribution first. Adapt the A2b integer
  mass head only if this is material; do not apply a raw continuous PPO density
  to a rounded pickup count.

**2026-09-24 A8 quantity follow-up.** Keep the promoted A8 scorer and the
categorical ALL market head. A factorwise intervention on 128 matched seeds per
opponent first suggested that sampling unit and market-kind choices while taking
greedy quantities improves sampled money. A fresh 256-seed-per-opponent
replication with identical executable source and artifact confirmed it:
`unit_kind` decoding scored **0.7168** against **0.6777** for all-head
sampling, a paired **+0.0391** score difference (exploratory 95% interval
**[+0.0176,+0.0625]**). Mean money increased **$2,451** (interval
**[$884,$4,151]**). The gain was concentrated against scripted v27
(+0.0742 score), while starter remained near saturation. These are the same
trained weights and seeds; the changed choice rule is the intervention.
`scripts/evaluate_architecture_campaign.py` now exposes `--decoding unit_kind`
for this reproducible stochastic evaluation. This does not change the
deterministic submission agent or PPO's temperature-1 behavior policy.

Quantity-only sampling with greedy unit and kind choices on 128 matched seeds
scored **0.9766** overall, versus **0.9922** for full argmax (paired
**−0.0156**, interval **[−0.0313,−0.0039]**) and lost **$3,432** mean money.
The amount head has a measurable sampling cost, but remains far stronger than
the rejected fraction heads; most of the all-sampled loss comes from the other
heads or their interaction. A new market amount representation has no outcome
case from this evidence alone.

The P1 audit covered 512 current-rules teacher episodes and 3,413,120 active
unit decisions. Pickups were **67,580** decisions (1.98%); **60,383** selected
amounts were below the legal maximum, mostly wheat and fertilizer. A separate
64-game sampled A8 self-play audit (128 player trajectories) found **16,382**
pickups, **14,775** below the legal maximum. A8's wheat and fertilizer amount
histograms remain concentrated at small integers. This is not a paired
teacher-policy comparison, but it shows A8 already executes the relevant
nonmaximum amounts. Its local affordance scorer includes action identity,
amount, held resource, and shed stock, so the existing categorical pickup
actions can represent conditional amount preferences. No pickup-head rewrite
is justified by frequency alone. Goose pickups had zero legal opportunities in
this A8 panel; the absent egg trade is upstream of pickup quantity.

The replication reports are
`artifacts/probes/unit-affordance-20260924/a8-qdecode-replication-{sampled-matched-source,greedy-quantity}.json`;
both record source identity `e24c45e00210cb0eb8608433f7cb76fc1c1f01657620518b3715cae984c0c7dd`.
The first 128-seed `unit_kind` diagnostic injected that mode at invocation
and its source manifest did not capture the injected code; use the 256-seed
replication for the decision. The pickup reports are
`pickup-amount-audit.json` and `a8-pickup-amount-onpolicy.json` in the same
probe directory. The canonical and frozen A8 snapshots have byte-identical
policy source (48 Python files) and native engine source (6 Rust files).
Cross-tree attempt **9882** matched all 512 canonical game rows, but its
Python import resolved to the editable canonical package despite running from
the frozen directory. It is not a cross-tree witness. Corrected job **9886**
set `PYTHONPATH` to the frozen snapshot's `src`; its recorded source identity
`b491a032b3dd7fe72b7be54e8568937cdffd18d451cc62c4f73781fb0f95c4ce`
equals the promoted artifact's. Against the canonical source on the same
256 seeds per opponent, all **512** sampled game rows and summaries matched
exactly. This witnesses canonical evaluation for this checkpoint and decode
mode; submission bundle export remains a separate validation step.
The paired decode jobs were **9858/9859** (discovery), **9866/9867/9874**
(replication, including the source-matched repeat), **9860** (quantity-only),
and **9872** (argmax); the pickup audits were **9863/9868** (teacher) and
**9873/9881** (A8, with source-hash refresh). GPU evaluations used exclusive
`mlq` jobs capped at two minutes.

### 4.4 Promotion and resource rule

Current LeJEPA action-run initializer is the promoted A8 full PPO checkpoint
`artifacts/promoted/unit-affordance-ppo-20260924.pt` (SHA-256
`6d36c49bc7d1e694a6d027b5a67f55abd95e16e53310cc4779a419a275c0a9cd`),
with `unit_affordance_scorer=true`. Keep promoted quantity ALL as the fixed
paired control for decoder claims; retain the A8 scorer in future action
arms unless an ablation explicitly removes it.

A clear winner in paired native outcome and its arm-specific behavior becomes
the initializer for later action runs immediately; record the source and
checkpoint digest. For market arms, behavior includes relevant trade volume.
A one-seed argmax gain or held-out BC NLL gain alone is a screening result.
Keep both decoding modes and the carrot/tomato/egg units in every
market-related panel. Training jobs stop at 30 minutes; benchmarks stop at
two minutes. Do not launch a new long run merely to obtain a benchmark
number that would take longer to compile than to measure.

### 4.5 Public teacher screen

The publicly callable [Kaito v48 notebook](https://www.kaggle.com/code/kaitofukami/40-40-early-floor-39-46-top-10-v48-fast-routes) and [Boatlee V20 notebook](https://www.kaggle.com/code/boatlee/v20-adaptive-r1-multi-route-agent) have complete agent source and historical Kaggle scores, but no paired 1.32.7 result against our v16 teacher. Their route tapes schedule small carrot sales, with no regular tomato/egg sales; Kaito has a possible final-turn liquidation path for all three if stock remains. [Boatlee V16-RC5](https://www.kaggle.com/code/boatlee/v16-rc5-high-score-8c-4s-premium-market-lead) schedules no trio sales. The [island-GA repository](https://github.com/destbreso/kaggriculture-island-ga) includes price-aware code but withholds winning schedules. These are source-code observations, not executed trade counts. A candidate enters the teacher pool only after its public source runs under 1.32.7, beats v16 on paired seeds, and demonstrates executed trio trades. No replacement is promoted yet.

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
   SELL_MILK/SELL_WOOL, PASS versus act) are when-to-act questions. If longer BC already removes most
   of them, interface gains will read small at BC.

## Appendix: reproducing the measurements

Scratch scripts, run with `CUDA_VISIBLE_DEVICES= .venv/bin/python -P <script> <corpus dirs>` under
the 16 GB memory scope. They are not part of the repository.

- `corpus_action_stats.py`: sections 1.3 and 1.5.
- `corpus_diversity.py`: section 1.2, duplicate kinds, and path axis order.
- `pointer_relabel_stats.py`: hindsight relabel legality, shared targets, PASS waits.
- `departure_mass.py <bc-actor.pt> <corpus dirs>`: section 1.4.
