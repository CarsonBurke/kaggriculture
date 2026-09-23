# Action-interface ablations, 2026-09-22 to 2026-09-30

Plan only. Nothing here has been implemented or trained. Every number below was measured for
this plan from the corpora, the code, or the shared BC checkpoint, and the scripts that produced
them are listed in the appendix. Final submission deadline: 2026-09-30 23:59 UTC. The plan freezes
the submission candidate at 2026-09-29 12:00 UTC so a full day is left for building and validating
the bundle.

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
   35% of mixed turns. Sells-first is still the right canonical order. Selling first adds money
   and shed room, with one exception: BUY_PRODUCT_WHEAT and BUY_PRODUCT_FERTILIZER add to the shed
   stock that SELL_WHEAT and SELL_FERTILIZER draw on
   ([core.rs:2779](rust/kagg_env/src/core.rs#L2779)). The corpus has 512 turns (one per episode)
   that sell wheat after buying it, and none of them becomes infeasible under sells-first. The A1
   round-trip gate checks this per turn instead of assuming it.
5. **"A production PPO iteration is about 6 s."** That is the lejepa family (6.4 s median over
   64 iterations, `runs/lejepa-full-20260922`). The production entity actor runs at 12.1 s median
   (`runs/structural-gae-20260918/component-control`, about 2.45 s rollout and 9.4 s update).
6. **"Market is interleaved unit by unit."** Within one slot both players are quoted the same price
   each unit round, so a slot's execution is symmetric. What orders across slots is slot index, and
   only for price-moving kinds (the nine SELLs and the two BUY_PRODUCTs).

## 2. Arms

Every arm keeps the entity trunk, the production PPO recipe of the day and the production league.
The new heads live in the shared head layer (`EntityActor._initialize_heads`,
[entity.py:596](src/kaggriculture/entity.py#L596)), so lejepa inherits them. Each arm is a
**versioned action interface** (`action_interface` in the model config, validated like
`observation_schema_version`), because the baseline has to keep running beside every arm during
the ablation. Old checkpoints load as interface 1.

| Arm | Change | Rust sampler | BC re-extraction | Replay / factor format | Effort |
| --- | --- | --- | --- | --- | --- |
| A0 | Production clone, 2 epochs (exists) | no | no | no | 0 |
| A0-12 | Production clone, 12 epochs (convergence control) | no | no | no | queue only |
| A1 canon | Canonicalized corpus, production heads | no | yes (CPU) | no | 0.5 day |
| A2 all | A1 plus an ALL encoding for quantities | small | no (reuses A1) | no | 1 day |
| A3 set | Market as a per-kind canonical set, ALL included | yes | yes (CPU) | yes (market arrays) | 3 days |
| A4 pointer | Target-tile pointer for units | yes | yes (CPU) | yes (unit arrays) | 3 days, gated |

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

**PPO.** Matched budget: 150 waves from each arm's own 12-epoch clone, 10 critic-warmup waves, the
production recipe of the day for every arm (if a drift-control recipe from the RL review lands
first, all arms use it), and 3 seeds per arm, because the loop is not reproducible under a fixed
seed ([REVIEW_RL_20260920.md](REVIEW_RL_20260920.md) section 1). Readouts: the external argmax panel
against v27 and starter at each committed checkpoint (existing `--external-eval`), paired v27 bank
change against the arm's own initializer, sampled starter score, and per-head entropy.

## 4. Stages, gates and schedule

### Stage 0 (2026-09-22 to 09-23 12:00 UTC): A0-12 and the diagnostics that order the arms

- **A0-12 clone.** The 12-epoch clone of the production interface, 2 seeds, trained first because
  0a, 0c and the A4 gate all read it. About 0.5 GPU-hour.

- **0a. Per-head departure cost.** Evaluate A0 and A0-12 on the v27 and starter panels, sampling
  one head family at T=1 and the rest at argmax: {none, all, units, kinds, quantities}. The native
  path already supports this without Rust changes. In `_stage_gpu_preferences`, Gumbel noise goes
  only on the chosen heads, and the Rust `deterministic` flag governs only quantities in the
  select path ([core.rs:2045](rust/kagg_env/src/core.rs#L2045)). The Python change is evaluation
  only, but it is more than one argument: the host `deterministic_rows` and the
  `gpu_deterministic_rows` are built from one array
  ([rollout.py:2344](src/kaggriculture/rollout.py#L2344),
  [rollout.py:2383](src/kaggriculture/rollout.py#L2383)) and must be split per head family; the
  per-head tensors must be allocated before graph capture; and the `learner_stochastic` check in
  [evaluate_architecture_campaign.py:200](scripts/evaluate_architecture_campaign.py#L200) becomes
  a per-head decoding argument. About 20 panels, **under 1 GPU-hour**.
- **0b. Teacher canonicalization loss.** Wrap public-v16 in an agent that rewrites only its market
  list: (i) merged, fixed order; (ii) merged, impact-ranked sells; (iii) unchanged, as control.
  Play it against public-v27 and against unmodified v16 in the official engine, 64 seeds, both
  seats. The teacher is open-loop (1.2), so its later actions do not react to the rewrite and the
  paired bank difference is the order's own worth. Paths need no test (1.1). **CPU only**, about an
  hour on 8 workers.
- **0c.** Recompute section 1.4 for A0-12 on the holdout seeds. (The section 1.4 numbers used the
  first four files per corpus, which are training seeds, and the fp32 Python encoder. Both biases
  are small on a 99.9%-deterministic corpus, but the gate numbers should be holdout numbers.) CPU,
  minutes.

**Gate after Stage 0.** Build A4 inside this window only if (a) unit-only sampling explains at
least a third of the sampled v27 gap (all-sampled minus argmax) in 0a for A0-12, and (b) a second
implementer can work in a separate worktree, since A3 and A4 both edit `sample_factors` and
`select_factors`. Otherwise A4 is written up and deferred. Choose the A1/A3 sell order from 0b:
fixed order unless impact order beats it by more than the paired interval. Add Plackett-Luce only
if both lose to the teacher's order by more than the interval.

### Stage 1 (09-23 12:00 to 09-24 18:00): A1 and A2 clones

12-epoch clones, 2 seeds each, under 2 GPU-hours with panels. Score is the mean match score
(win 1, draw 0.5, loss 0) from the evaluation script. **Gate (applies to every BC arm, here and
in Stage 2).** (i) The argmax v27 score is no more than 2 percentage points below A0-12's, and the
argmax paired-bank interval includes zero or better. (ii) Expected departures per game fall by at
least 30%, **or** the sampled v27 paired-bank interval excludes zero on the positive side. Arms
passing both enter the stack; the best passing Stage-1 arm is the fallback submission base.

### Stage 2 (09-24 18:00 to 09-27 18:00): A3, and A4 if gated

Implementation with parity tests first: Rust-vs-Python masks and likelihoods, the null round trip
on the canonicalized corpus with engine replay, and replay parity at production shape. Then clones
and panels under the Stage-1 gate, about 1 GPU-hour. The hard cut is 09-27 18:00 UTC, which is the
three-day estimate with no slack: if A3 is not passing its parity tests then, it is cut. A4 runs in
parallel only with a second implementer in a separate worktree.

### Stage 3: matched-budget PPO, in two waves

- **3a (09-25 to 09-26), while A3 is being built.** Baseline A0-12 and the passing Stage-1 arms,
  3 seeds times 150 waves each. The GPU is otherwise idle during A3 implementation, so this costs
  no calendar time.
- **3b (09-28), only if A3 passes Stage 2.** A3, 3 seeds, the same recipe (with the recalibration
  in 2.3 risk 4 if triggered), compared against the 3a baseline runs.

At 12.1 s per wave a run is 30 minutes. The total is at most 12 runs, about 6 GPU-hours plus about
2 hours of checkpoint panels; half that on lejepa. **Decision rule.** Each seed's wave-150
checkpoint plays the 256-map argmax v27 panel. The statistic is the paired v27 bank change against
the arm's own clone, and the comparison with the baseline uses a bootstrap that resamples maps and,
within each, seeds. An arm wins if the 95% interval of (arm minus baseline) excludes zero on the
positive side and no seed's argmax v27 score falls more than 5 percentage points below its own
clone's. Three seeds are too few for a seed-level test; the map-paired bootstrap carries most of
the power, and the result is still read as evidence, not proof. If no arm wins, PPO has not beaten BC under either interface,
which is the current state of every recipe. The submission is then chosen among BC clones by the
argmax panels, and an interface arm is preferred only if it passed the Stage-1 gate with a better
argmax bank.

### Stage 4 (09-28 18:00 to 09-29 12:00 UTC): final candidates

Official-engine v16 panel for the top two candidates, the argmax v27 and starter panels,
`build_submission` and `validate_submission`. Freeze at 12:00 UTC on 09-29; the remaining 36 hours
are buffer for bundle problems only.

### Budget

| Stage | GPU-hours | Notes |
| --- | ---: | --- |
| 0 | < 1 | plus about 1 CPU-hour of official-engine games |
| 1 | about 2 | BC is 8-15 s per epoch on the entity actor |
| 2 | about 1 | |
| 3 | about 8 | up to 12 PPO runs plus panels; about 4 on lejepa |
| 4 | about 1 | plus 2-4 CPU-hours for the v16 panels |
| Total | about 13 | Implementation time, not GPU, is the binding constraint |

The critical path is Stage 0 (0.5 day), A1 and A2 (1.5 days), A3 (3 days, ending 09-27 18:00),
A3 PPO (about 1 day including panels) and the final panels (0.75 day), ending at the 09-29 12:00
freeze with no slack. The Stage-1 arms' PPO runs overlap A3 implementation. That is why A3 has a
hard cut, why A2 plus the 3a results are the fallback deliverable, and why A4 needs a parallel
implementer or waits.

## 5. Biggest uncertainties

1. **Whether any interface fixes what PPO is losing.** The review attributes the PPO erosion
   mostly to objective, credit and drift. Interface arms can shrink the sampled tail and still not
   make PPO improve over BC. Stage 3 is designed to show that honestly, not to rescue it.
2. **A3's departure count.** More, easier decisions per turn could raise T=1 departures. The Stage-2
   gate measures this before any PPO is spent.
3. **Sell-order value against v27.** Unknown until Stage 0b. It decides between fixed order,
   impact order, and Plackett-Luce.
4. **Timing uncertainty is untouched by any interface.** The largest departure flows (STOP versus
   SELL_MILK/SELL_WOOL, PASS versus act) are when-to-act questions. If A0-12 already removes most
   of them, interface gains will read small at BC.

## Appendix: reproducing the measurements

Scratch scripts, run with `CUDA_VISIBLE_DEVICES= .venv/bin/python -P <script> <corpus dirs>` under
the 16 GB memory scope. They are not part of the repository.

- `corpus_action_stats.py`: sections 1.3 and 1.5.
- `corpus_diversity.py`: section 1.2, duplicate kinds, and path axis order.
- `pointer_relabel_stats.py`: hindsight relabel legality, shared targets, PASS waits.
- `departure_mass.py <bc-actor.pt> <corpus dirs>`: section 1.4.
