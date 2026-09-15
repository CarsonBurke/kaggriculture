# kraggiculture

Research and evaluation tooling for the Kaggriculture simulation competition.

Production training is BC-initialized self-play PPO with DAPO's asymmetric clip
band, discount-correct potential shaping, and full-return actor/critic GAE. A fresh
production run must load one behavior-cloned actor, then fits its fresh critic
for at least ten iterations and until every member's previous fresh-wave
pre-update Monte Carlo-return R-squared reaches 0.10. Only an existing
checkpoint can bypass that initialization. An exact batched Rust simulator supplies
high-throughput rollouts; the pinned Kaggle environment remains the parity
oracle and final evaluator.

The default critic uses categorical HL-Gauss cross-entropy with 255 linearly
spaced, exactly mirrored bins on `[-2.2, 2.2]`, tail-stable Gaussian mass
calculations, FP32 capped logits, and a paired expectation reduction.
`--value-sigma-ratio` tunes Gaussian sigma in bin widths (default `3.0`,
or `0.05197` in return units). The former `0.75` setting gives sigma `0.01299`.
`--scalar-value true` selects the unclipped scalar half-squared-error ablation;
the value-support and smoothing fields are inert in that mode.
Smoothing is independent of the support and may differ from an actor-only BC
checkpoint; a training resume still requires identical model configuration.
The [HL-Gauss paper](https://arxiv.org/pdf/2403.03950), section 5.1.2, motivates
tuning return-space bandwidth rather than assuming the same ratio remains
appropriate after changing bin count. Compare rollout value accuracy and policy
performance, not raw cross-entropy across smoothing settings: the interior
label-entropy floor rises from about `1.2003` to `2.5222` nats for those ratios.
The categorical arm uses neither symlog nor a critic EMA.

Install development and training dependencies, then run the CPU-safe default
validation path:

```bash
uv sync --extra dev --extra train
uv run pytest -m "not cuda"
```

CUDA tests carry the `cuda` marker and must use the machine-wide ML queue:

```bash
mlq submit --name kagg-cuda-tests --max-parallel-runs 1 --priority 0 \
  --time-limit 30m -- uv run pytest -m cuda
```

The first Rust-backed rollout automatically builds the native extension with
`cargo build --release`. Rust traces must pass differential tests against
`kaggle-environments==1.32.6` before they are admitted to training.

Run the complete native correctness gate with:

```bash
cargo test --manifest-path rust/kagg_env/Cargo.toml
cargo clippy --manifest-path rust/kagg_env/Cargo.toml --all-targets -- -D warnings
cargo build --manifest-path rust/kagg_env/Cargo.toml --release --lib
uv run python rust/kagg_env/tests/parity_oracle.py --games 8 --steps 719
uv run python rust/kagg_env/tests/binding_safety.py
```

The parity oracle compares every public and private game field, encoded model
input, and shaping potential after every transition. It is intentionally the
release gate for simulator changes, not a statistical approximation. The
binding safety gate additionally proves that malformed, strided, read-only, or
aliased NumPy buffers fail before simulator state can advance.

Measure the scalar-core batch engine independently of model inference with:

```bash
cargo run --manifest-path rust/kagg_env/Cargo.toml \
  --release --bin bench_env -- 4096
```

GPU training and throughput measurements should be submitted through the local
ML queue so they do not contend with another experiment. Freeze the complete
Python/Rust/build input tree first; all DAG nodes must use that read-only tree,
the frozen `src` on `PYTHONPATH`, and a writable digest-keyed Cargo target:

Executable defaults are the configuration guide. Use each entrypoint's `--help`
and inherit its settings; `src/kaggriculture/production.py` owns production model
and PPO defaults. For BC, select `--production-model` and provide the corpora and
output directory without copying an epoch or optimizer recipe from `RUNS.md`.
That file records experiments and historical evidence, not competing defaults.

Every path below is anchored to the repository root rather than to `$PWD`, so
the recipe does the same thing from any working directory. That is not
cosmetic: with `$PWD`, running it from inside the crate points
`CARGO_TARGET_DIR` at `rust/kagg_env/artifacts/cargo-target`, which buries a
full build tree inside a source root, where `source_identity` must then decide
whether it is source or output.

```bash
repo=$(git rev-parse --show-toplevel)

snapshot=$("$repo/.venv/bin/python" "$repo/scripts/freeze_source.py" | \
  "$repo/.venv/bin/python" -c 'import json,sys; print(json.load(sys.stdin)["source_root"])')
digest=${snapshot##*/}

benchmark() {
  name=$1
  shift
  mlq submit --name "kagg-ppo-$name" --max-parallel-runs 1 \
    --priority 0 --time-limit 30m --cwd "$snapshot" \
    --env PYTHONPATH="$snapshot/src" --env PYTHONDONTWRITEBYTECODE=1 \
    --env CARGO_TARGET_DIR="$repo/artifacts/cargo-target/$digest" -- \
    "$repo/.venv/bin/python" scripts/benchmark_ppo_iteration.py \
    "$@" \
    --output "$repo/artifacts/benchmarks/$name-ppo.jsonl"
}

benchmark eager    --rollout-forward-mode eager    --rollout-bfloat16 --update-compile-mode eager
benchmark mixed    --rollout-forward-mode eager    --rollout-bfloat16 --update-compile-mode default
benchmark compiled --rollout-forward-mode graph    --rollout-bfloat16 --update-compile-mode default
```

All three time the production batch from the same source snapshot and differ
only in the knob each step turns on. Neither knob is a boolean: each names the
execution mode of its phase, and `eager` is one of those modes rather than the
absence of a choice, so the chain's first node states `eager` on both sides and
each step moves one of them to a compiling mode. The collection precision is
stated on every node rather than left to the default because it is not a knob:
the launcher requires it to be identical across the chain and equal to
production's, so a chain measured in fp32 is rejected instead of launched.
Launch training from the complete set through the same queue and frozen source:

```bash
mlq submit --name kagg-ppo-training --max-parallel-runs 1 --priority 0 \
  --time-limit 168h --cwd "$snapshot" \
  --env PYTHONPATH="$snapshot/src" --env PYTHONDONTWRITEBYTECODE=1 \
  --env CARGO_TARGET_DIR="$repo/artifacts/cargo-target/$digest" -- \
  "$repo/.venv/bin/python" scripts/launch_calibrated_training.py \
  --eager-report "$repo/artifacts/benchmarks/eager-ppo.jsonl" \
  --mixed-report "$repo/artifacts/benchmarks/mixed-ppo.jsonl" \
  --compiled-report "$repo/artifacts/benchmarks/compiled-ppo.jsonl" \
  --init-actor-from "$repo/runs/schema3-bc/bc-actor.pt" \
  --run-dir "$repo/runs/ppo-main"
```

The launcher rejects partial reports or any mismatch in source, hardware, seed,
precision, architecture, model, PPO, or data-generation settings. An unflagged
benchmark uses the exact structured production model.

Compilation is decided per knob, not per run. The rollout collector and the
update are timed separately and a device synchronization ends each, so the
three phases add to that iteration's total exactly, and the launcher checks
that identity on every iteration of every report. The reports form a chain from
all-eager to all-compiled in which each step turns exactly one knob on and
changes nothing else; every report declares the configuration it ran under, so
the chain's shape is evidence rather than an argument. Two knobs therefore need
three reports, and the middle one is what makes the decision attributable.
Differencing all-eager against all-compiled moves both knobs at once, so a
per-phase ratio taken across that pair carries whatever between-run drift it
happened to have. That is not hypothetical. On the conv model the rollout
phase measured 8.5054 s in the all-eager run and 8.0847 s in a run that also
left the rollout uncompiled -- 4.9% apart with the knob unchanged -- while the
two-report differencing credited the rollout knob itself with 8.0%. Against the
report that isolates it the knob measures 1.027 and loses; against the
contaminated pair it measured 1.080 and won.

Each knob is scored by what it does to the whole iteration rather than to its
own phase: the steady total median that the node before the step actually
measured, divided by that same total with only the knob's own phase median
replaced by what the node after the step measured for it. Both ends are
therefore anchored on an iteration the benchmark ran rather than on a budget
assembled by summing phase medians. Substituting one median into another is
only meaningful while a summary's phase medians describe the iterations its
total was taken from, so the launcher also rejects any batch summary whose
three steady phase medians sum more than 1% away from that summary's own
steady total median. The per-iteration identity does not imply that: a median
of sums is not a sum of medians, and a budget assembled from three different
iterations describes none of them. Matched reports measure hundredths of a
percent against that bound. The attributed ratio has to clear 1.05. A
per-phase ratio would flatter a knob whose phase is a small share of the
iteration, and compilation is not free -- it costs warmup, replay divergence
against the collector, and a decision that is stamped into provenance and
cannot be revised mid-run. Knobs are enabled as a prefix of the chain, so the
decided configuration is a node of the chain and was therefore measured rather
than projected; a knob that clears the floor only on top of a knob that does
not is refused outright, because the chain never measured it in the
configuration that would actually run.

The two knobs are not a formality: at six repeats on the conv model they land
on opposite sides of the threshold. The update knob is worth about 2.73x on the
whole iteration and is enabled. The boolean rollout knob these numbers come
from measured about 1.006x on the whole iteration -- 1.027x on its own phase,
roughly 0.2 s out of 37 s -- and was rejected. An earlier two-repeat
calibration reported the rollout at about 0.46x, the collector's per-step graph
replay supposedly costing more than the kernel launches it removes; that figure
is withdrawn rather than explained. It
does not survive six repeats, where the compiled rollout median is 7.876 s
against an eager 8.200 s in that earlier run and 8.505 s in the current one. A
single blended total would still enable both knobs, since their sum favors
compiling, and the run would carry the rollout's warmup and replay risk to buy
two tenths of a second. Read each run's own decision file for its numbers
rather than these; the point that survives is the shape, not the magnitude.

That rejection was right about the mode it was offered, and it is why the
rollout knob is a mode rather than a flag. The boolean's only "on" value was
`cudagraphs`, which measures 5.309 ms against eager's 4.907 ms on the isolated
collection forward, median of 60 waves in fp32: slower than not compiling at
all, so no chain over that flag could have found anything better than eager.
`inductor` with `reduce-overhead` measures 2.720 ms in fp32 and 1.626 ms under
bf16 autocast, and end to end on the stage profile it is 1.87x faster -- 9.408
ms per step against 5.035 ms, a projected rollout phase of 6.76 s against
3.62 s. These are historical throughput measurements, not evidence for the
current numerical contract; source-bound calibration must be regenerated after
changing operators or compiler settings.

PPO clips each active conditional action ratio independently. By default, policy
loss sums active component surrogates within each state and averages over valid
states. KL, clipping statistics, and entropy remain means over genuine active
components, not means of per-state means. The stopping threshold remains
`target_kl = 0.03`.
Static minibatch padding has zero
loss/gradient/metric weight, and every epoch gets a fresh permutation. Auxiliary
trajectory plans follow that permutation and exclude padding. The native horizon
of 720 states produces 719 transitions; pre-step observations/actions and
post-step rewards remain aligned, with zero bootstrap at true episode termination.

Rollout and update use the same FP32 master parameters under compiled BF16
autocast, rather than sampling from a separately rounded parameter replica.
Stored behavior likelihoods are the sampler's actual probabilities, never
replaced by update replay. The small selected-kind quantity head uses the same
FP32 accumulation order as the native sampler. CUDA RMS normalization packs
independent rows into width-specific tiles while retaining batch-independent
FP32 row arithmetic. Conditioning means have a fixed token reduction order,
and linear bias addition has an explicit BF16 rounding boundary. Structured
attention uses one memory-efficient CUDA SDPA backend across batch sizes and
autograd modes, with head padding and GQA groups folded into the query axis
instead of duplicating K/V tensors. Policy compilation preserves cast boundaries
and disables inference-only pattern rewrites; BF16 matrix products retain FP32
accumulation.

`update_replay_joint_kl` measures the complete action likelihood as a diagnostic;
`update_replay_component_kl` measures the component-averaged trust-region quantity.
Replay audits compare both inference and gradient graphs against unchanged sampler values.
Numerical parity is necessary for PPO correctness, not evidence of better returns.

All three reports time the production batch and nothing else, because the
decision reads the steady medians at 128 games and nothing from the other
sizes. Sweeping four batch sizes to produce them cost about four times the
calibration. Shorten all three rather than some of them -- the launcher
compares the sweeps, so an uneven set would differ in protocol as well as in
mode. `--repeats` is part of that comparison: it is recorded in the
configuration record and the launcher refuses a chain that disagrees on it, so
whatever you choose has to be passed to every report. Choose six. Compilation
has warmup that eager does not, and two repeats leave a single steady
iteration, which cannot separate a warming phase from a steady one on the side
where that distinction decides a knob -- which is how the withdrawn rollout
number came about. Six is cheap -- minutes -- and on the eager side it put the
steady update within 0.15% and moved the steady total median 0.24% against the
two-repeat value, so it costs nothing in fidelity to run it on every node of
the chain.
Sweep with `--games 64,112,128,256` when the question is scaling or memory
headroom, which is a separate study from this one.

For VRAM comparisons, read `peak_cuda_reserved_bytes` alongside
`peak_cuda_bytes` (peak live allocations). The iteration records also expose
`current_cuda_allocated_bytes` and `current_cuda_reserved_bytes` after the
update, so retained tensors can be distinguished from allocator caches.
PPO runs actor and critic updates on the caller's CUDA stream so both branches
reuse one activation allocation pool. Even a stable pair of separate streams
strands each branch's cached blocks: at production shape this forced repeated
allocator eviction and remapping despite much lower live memory. Ordered
execution removes that overhead without changing precision, batch size, or
objectives; do not replace it with per-iteration `empty_cache()`, which discards
the working set.

The production default is an HL-Gauss critic with **actor NextLat disabled** and
plain-sum critic training: value loss plus coefficient-1 latent SmoothL1 and
coefficient-1 decoded-value loss, at horizon 1. Critic source gradients are not
norm-matched. For a scalar critic, decoded-value KL is unit-variance Gaussian KL
(half squared mean error), not a degenerate one-category softmax. Distributional
critics use categorical decoded KL over the same capped logits as their readout.

Actor and critic use separate backbones and latent banks, with no shared
parameters. The value loss remains attached through the critic's value decoder
to its latent bottleneck and ViT; it never updates the actor backbone. Critic
NextLat also trains its source backbone and predictor, but detaches its
successor teacher and the value-head weights used for auxiliary decoding.
Its categorical KL is teacher-to-student over the full value distribution.

Actor NextLat is opt-in: set `--structured-latent-coefficient 1` and
`--structured-decision-coefficient 1`. `ActorDynamics` jointly predicts the
normalized unit and market decision representations immediately before the final
policy projections, after the entity decoders. Its input is these representations
and the joint action; inactive unit actions and post-STOP market suffixes are
masked. Latent SmoothL1 averages over valid successor coordinates, not separate
family means. Unit targets require cumulative survival across the prediction
horizon; newborn successor units are excluded. Reached successor market orders
participate in the loss.

Decoded teacher-to-student KL uses only the **frozen final policy projections**,
not a replay of the entity decoder. Cached, detached successor head-input
representations supply the teacher. Student and teacher projections use identical
FP32 arithmetic with successor legality and activity masks. The KL pools active
unit, market-kind, and market-quantity decisions. Setting both actor coefficients
to `0.3333333333333333` divides the complete auxiliary objective by three policy
head families; it does not replace the internal coordinate or decision means.

PPO and the auxiliary share one actor forward and one additive combined backward.
Auxiliary gradients enter the live source head-input representations and flow
through their entity decoders and actor trunk; successor targets and auxiliary
readout weights are detached. Normal PPO gradients still train the final policy
projections. There is no actor source-gradient balancing.

The auxiliary-enabled actor trunk uses non-reentrant activation checkpointing:
backward recomputes its intermediates without splitting the PPO minibatch or
adding optimizer steps/backward calls.

This configuration remains experimental: contract tests establish gradient and
execution correctness, not improved learning. Actor/BC parameter keys are
unchanged, but predictor states from different attachment architectures are not
interchangeable. Warmup release still measures critic readiness, not predictor
readiness; persistence scores remain diagnostic only.

Both actor coefficients default to zero, so no actor predictor or predictor
optimizer is constructed. `--structured-critic-gradient-balance` remains an
experimental opt-in for critic-only 50/50 source-cotangent norm matching; the
default is `--no-structured-critic-gradient-balance`.

`--policy-loss-reduction states` is the default. With the default
`--policy-ratio-scope components`, each component ratio is clipped independently,
then the summed surrogate is divided by valid states rather than active
components. `--policy-loss-reduction components` retains the former control
reduction. Padded rows contribute neither loss nor denominator. KL, entropy, and
clipping diagnostics retain their component-normalized units.

`--policy-ratio-scope joint` requires state reduction. It sums active conditional
action log-ratios per state, applies one PPO clip to the resulting joint ratio,
and uses state-mean joint KL for the trust-region stop. The configured KL threshold
is unchanged, so this is a tighter trust-region experiment, not a calibrated
equivalent of component clipping. Entropy and `component_kl` remain
component-normalized; sampler parity always uses component KL.

PPO `approx_kl` uses the sampled-action estimator
`exp(log_ratio) - 1 - log_ratio`, where `log_ratio = log_pi_new - log_pi_old`,
at the selected ratio scope. It is not the full categorical KL used by NextLat.

`--per-entity-critic true` adds centralized value predictions for owned units and
market orders alongside the global value. Active entity advantages are normalized
over owned unit/order entries; market-kind and quantity decisions share their
order's advantage. All predictions use the same team return target, with primary
critic loss averaged over active global/entity slots within each state. Global
critic diagnostics and critic NextLat retain the global value representation.
This experiment requires both GAE lambdas to be one and component ratio scope:
pass `--actor-gae-lambda 1` explicitly to override the promoted VAPO default.
It cannot be combined with a shorter actor trace or joint ratios.

PPO has no patch, economy, or opponent-state prediction objectives. Its actor
predictor reads unit/market head-input representations and actions; its critic
predictor reads the value representation and actions. Existing BC-only world-feature experiments
remain separate from this PPO contract; they are not evidence for a world model.
When actor NextLat is enabled, critic-warmup and KL-stop phases still freeze the
actor while fitting its predictor. Fresh-wave persistence scores are diagnostic
only.

PPO exports the unit and market head-input representations across its compiled
actor boundary; BC retains the full world-belief interface. Frozen
actor predictor training uses a cached compiled BF16 belief-only forward,
without unused policy logits. A released-actor backward warmup is discarded once
per callable/configuration/shape, not once per frozen wave.

Rollout statistics validate categorical support on existing host masks, avoiding
two device-to-host boolean barriers per environment step. Unit/kind statistics
and entropy packing are compiled; native quantity likelihoods no longer make an
unnecessary GPU roundtrip. Built-in league agents occupy no neural ensemble
slots. Compiled neural inference rounds lane counts and per-lane widths up to
powers of two, reusing fewer compiled layouts as opponent assignments change.
Extra rows and lanes duplicate valid inputs/weights and are discarded before
sampling; opponent selection, physical games, and training rows are unchanged.
Only encountered buckets compile, not every reachable layout in advance.
The compile guard tracks these physical buckets rather than raw assignment counts.
Mutable ensemble weights are thread-owned. Native paired encoding computes each
physical farm's public tile features once and reuses them for the opposite seat.
The source-bound 2026-09-12 probe in
`artifacts/probes/balanced-objectives-20260912/summary.json` measured cached
production-shape waves at 33.15 → 25.51 seconds (23% less time): rollout
6.77 → 5.69 seconds, update 26.37 → 19.82 seconds. Each arm used one initial
wave plus two cached repeats, 230,080 states, BF16, 4,800-state minibatches,
128 self-play games and 64 league games, and the trainer's expandable allocator
and CPU-thread settings. All 48 actor, critic, and predictor minibatches ran.
This comparison includes the parameter-gradient to source-cotangent balancing
change; it is not an identical-objective optimizer A/B. The isolated four-optimizer
step with identical production-shaped gradients measured 14.51 → 11.38 ms median.
Allocator retries remained in both arms; neither full GPU utilization nor a
learning-quality improvement is established by these timings.

The 2026-09-13 update-memory probe
(`artifacts/probes/update-memory-20260913/summary.json`) isolates the remaining
allocator bottleneck and measures the optimized kernels at the same production
shape. Cached update times were 16.79 s at 4800 rows before these changes,
10.18 s at 4800 afterward, and 9.32 s at 6400 (36 minibatches instead of 48).
At fixed batch size, peak live/reserved VRAM fell from 20.27/26.56 GiB to
17.14/17.85 GiB; the 6400-row run used 21.12/22.09 GiB. Allocator retries fell
from 95 per wave to zero. Each timing is one unprofiled cached wave; profiled
repeats are excluded. Both completed 6400-row waves retained every state and
accepted all 36 actor, critic, and predictor steps. Its three-minute cap stopped
the optional third replay audit, not either measured wave. These are execution
measurements, not evidence that the larger-batch learning dynamics are better.

Contiguous validity segments receive independent random partition phases before
their bounded runs are shuffled. Every valid state appears once in the primary
epoch; auxiliary transition subsampling no longer aliases daily rollovers.
CPU-derived successor plans require every intervening step to belong to the
same contiguous trajectory, then compact eligible sources into bounded aligned
shapes. Padding contributes neither loss nor gradient. Critic value KL reduces
its singleton token dimension before masking rows, avoiding cross-batch
broadcasting.

Diagnostic iterations observe auxiliary source-belief cotangents through
zero-copy branch views during that same backward. Ordinary and observed calls
use identical view layouts; only the scalar-reduction hooks are conditional.
Preupdate and persistence scalars are packed for one final diagnostic readback.
There are no extra diagnostic backwards or retained-graph compiler variants;
ordinary buffer donation remains enabled. Captured rollout forwards include
fixed-index scatter and are submitted before CPU trajectory storage to overlap
device work with host copies.
Minibatch inputs and returned beliefs are released after their final use,
before the next gather/forward. Compact NextLat plans use spare occupancy-shape
slots to reduce padding while retaining every former
bucket boundary: padding never increases and there are still at most eight
aligned shapes per minibatch size.

Training uses discount-correct, exactly zero-sum potential shaping. Let `L[i,t]`
be player `i`'s actual liquid assets: bank money plus the exact proceeds from
selling every held product at the current market curve. With the game-defined
starting bank `k = 3000`,

```
P[t] = (L[0,t] - L[1,t]) / (L[0,t] + L[1,t] + 2*k)  # nonterminal potential
U[T] = (bank[0,T] - bank[1,T]) / (bank[0,T] + bank[1,T] + 2*k)  # terminal utility

r[0,t] = gamma * P[t+1] - P[t]  # nonterminal
r[0,T-1] = U[T] - P[T-1]        # terminal; terminal shaping potential is zero
r[1,t] = -r[0,t]
```

The production discount is `gamma = 1`. From the symmetric initial state
`P[0] = 0`, the complete shaped return is exactly the final-bank margin `U[T]`,
up to binary32 accumulation error. Intermediate potential differences cancel;
there is no early-lead or time-average occupancy objective. Every transition,
including the terminal transition, sums to exactly zero. Explicit gamma
overrides remain discount-correct: the complete discounted return becomes
`gamma^(T-1) * U[T]` at the fixed episode horizon.

Potential and terminal utility use the same bounded, zero-sum margin function.
The `2*k` denominator regularizes the slope near ruin: `3000` versus `0`
scores `1/3` for both cash-only potential and terminal utility. Equal banks score
zero, including mutual bankruptcy. At unchanged holdings, the terminal
correction is the difference between bank-only and liquidation margins, so
unsold goods lose their shaping credit rather than a dominant cash lead paying
a logarithmic scale-mismatch penalty.

Liquid assets deliberately exclude seeds, animals, planted crops, pending
yields, and land because the market cannot liquidate them. Market products are
valued by walking the engine's sell arithmetic unit by unit, including its
price-floor restock rule. Moving those products into the bank is therefore
potential-neutral, so cycling inventory cannot manufacture reward.

Rust supplies binary32 potentials and terminal utility; one Python reward
implementation applies the same configurable gamma to native and interpreted
rollouts.

`--reward-mode terminal-bank` removes shaping: nonterminal rewards are exactly
zero and each terminal transition pays the same signed final-bank margin used by
the shaped mode. It does not switch to binary win/loss or raw money. Collection
uses the actual terminal utility directly, and credit diagnostics omit the
potential correction in this mode. Rollout composition rejects mixed reward
modes; exact training resume requires the recorded reward mode to match.

The default `--reward-mode terminal-outcome` pays zero before termination, then
**+1 for a win, -1 for a loss, and 0 for a draw** instead of a final-bank margin.
Win/draw comparisons follow the official floating-point bank scores. Native
collection takes the sign of the terminal utility rather than comparing the
separately rounded binary32 bank telemetry, which can turn close wins into ties.
Credit diagnostics use the stored terminal outcome, without potential correction.
Explicit `--reward-mode shaped` and `--reward-mode terminal-bank` remain available.

The promoted VAPO temporal defaults are `--gamma 1`,
`--actor-gae-lambda 0.972183588317107`, and `--critic-gae-lambda 1`.
The actor value is `1 - 1 / (0.05 * 719)`, using VAPO's alpha `0.05` and the
full game's 719 transitions. It is fixed for this game, not adapted per batch.

The critic fits full, undiscounted Monte Carlo returns: the default target at
every valid state is the terminal win/loss/draw outcome. With explicit shaped
reward, the target is instead `U[T] - P[t]`. Actor advantages use a shorter GAE
trace, with geometric weight sum approximately 35.95 transitions,
to reduce variance while relying on the critic for longer-term value. This is
not a 36-turn planning cutoff; inaccurate critic predictions can bias the actor.
Collection shaping, advantages, and value targets share gamma. Targets outside
categorical support saturate at the outer atom, with the saturated fraction
reported.

The temporal settings were first promoted from dense-reward trial **7010**.
The user subsequently selected **7122**, the HL-Gauss VAPO terminal-outcome LR3
trial, as the new production default: terminal win/loss/draw reward and tripled
actor/critic rates, retaining HL-Gauss, component PPO clipping/KL, architecture,
and the existing auxiliary recipe. This adopts VAPO's temporal settings, not
every component of its training recipe. New launches inherit the new defaults;
explicit overrides and previously frozen commands retain their declared settings.

The default trust region is `target_kl = 0.03` on the active-component mean KL.
Its historical calibration does not establish the stopping frequency after
changing rewards and auxiliary balance; measure accepted minibatches explicitly.

Entropy is measured but not optimized. The main actor objective is clipped PPO;
production leaves the actor future-policy auxiliary off unless explicitly enabled.
The critic jointly optimizes one-step latent and decoded-value prediction
auxiliaries. Production uses one learner with 128 self-play games and 64 league games per wave (320
learner trajectories). Stale matchup evidence for built-ins and snapshots decays
toward 0.5 alike, so formerly easy opponents can become contested again.

With `inductor_graph`, training precompiles balanced league layouts up to the
configured lane budget on its first nonempty league wave. This moves their cold
compilation to startup without changing assignments or padding steady waves.
CUDA graphs remain wave-owned; new update-gradient phases can still compile
separately. Structured fused-MLP predictors refresh cached projections before
training and after each predictor optimizer step, including critic warmup.

Optional population training uses uniform ordered round-robin pairings and both
seats' trajectories. `--population 4` requires `--league-games 0`; frozen
snapshots and built-ins are excluded from population waves.
Population members receive distinct game seeds (`seed_start + g`), so each
member's games within a wave cover different maps except for direct
head-to-heads. The seeded pairing permutation changes which ordered pair owns
each map stratum across waves. Convolutional populations additionally cycle
identity, horizontal mirror, vertical mirror, and 180-degree frames; the
orientation transforms the encoded board, unit positions, and movement actions
consistently. The structured encoding used by production has no equivalent
orientation transform yet, so structured population waves use identity frames
rather than rejecting an otherwise valid run. Rollout batches retain row
orientations only for PPO replay; evaluation and submission use the real-board
identity frame.

Production architecture and PPO settings are selected explicitly by the shared
factories in `src/kaggriculture/production.py`. Production launchers serialize the
resolved configuration into each run's provenance; those records describe what
ran, while the executable defaults determine future launches.

Structured observation schema v3 includes per-unit carried-item insertion ranks,
alongside exact counts: DROP fills available shed space in that order and discards
overflow. Ranks are encoded consistently in Python/native storage for both players;
opponent inventory ranks remain critic-only. Separate goose/cow/sheep purchase,
shed and carried-stock tokens and public farmer/hand occupancy remain unchanged.
Rebuild native encoding and BC caches
and train fresh actors: old structured model artifacts are rejected, not migrated.

Fresh-wave persistence diagnostics compare each active loss with no-change
prediction through the same encoder/readout. A zero baseline is uninformative,
not evidence of success. Ratios never enable or disable representation learning.
Predictor fitting and preupdate diagnostics have separate synchronized timings.
Recovery checkpoint format 17 separates the bounded-margin reward and balanced
source-gradient regime from prior critic targets and optimizer moments. Older
containers, including version 16, remain actor-readable when their observation
schema matches, but are not resumable training states under the new objective.

`credit_preupdate_*` reports critic error against the rollout's terminal utility,
grouped by opponent and time-to-go. Default outcome diagnostics use the stored
terminal reward; shaped-mode diagnostics remove the known shaping potential and
include a potential-only baseline. High shaped-return explained variance alone
is not evidence of long-horizon prediction. Every 25 iterations, gradient diagnostics report
`structured_gradient_source_norm` and `structured_critic_gradient_source_norm`:
the actor head-input and critic value-belief raw auxiliary cotangent norms,
respectively. They are not parameter-gradient norms or main/auxiliary cosine
estimates. Observation does not change optimizer updates.

Fresh production training must be initialized from a BC actor through
`--init-actor-from`. The actor enters RL with a fresh critic and optimizers, no
persistent BC or KL term, and a critic-only warmup defaulting to a minimum of ten
iterations (`--critic-warmup-iterations 10`). Actor updates begin only after
that floor and after every member's previous fresh-wave
pre-update Monte Carlo-return R-squared reaches 0.10; failure to reach
that gate by iteration 40 stops the run instead of training against an unready
baseline. Both production launchers reject a fresh random actor; `--resume`
remains valid for continuing a checkpoint. Raw `train_ppo.py` remains available
for controlled from-scratch experiments.
Readiness uses `1 - MSE(G - V) / Var(G)`, not centered residual variance, so
constant value bias cannot disappear from the gate. Centered explained variance
remains separate telemetry.

Categorical CPU/native sampling accumulates positive unnormalized masses in
float64 and uses strict intervals. Rounding fallback selects only positive mass;
selected log-probabilities come from logits and the normalizer, without flooring
underflowed probabilities.

Actor and critic trunk base learning rates both default to `1.5e-4` (NorMuon
matrices), with `5.25e-5` for their ordinary Adam parameter groups. Production
and the direct training CLI default the separate value-head Adam LR to `4.375e-4`, preserving
its `25/3` boost over ordinary Adam groups. The raw training CLI accepts
`--critic-head-lr` as an optional absolute override. Each group retains its own
32-optimizer-step linear LR warmup and checkpointed state. NextLat predictors
inherit their corresponding actor/critic base rate unless explicitly overridden.
Embedding weights are assigned to Adam by module ownership, including tied
weights; direct learned latent/opponent/value queries also use Adam. Hidden
projection matrices remain on NorMuon. This corrects older structured-model
partitions that treated categorical tables and those queries as hidden matrices.
Model weight formats are unchanged, but old optimizer histories cannot be loaded
into the corrected partition: NorMuon state does not contain Adam's second-moment
history. Retain the original source for an exact historical resume; adopting the
correction requires fresh optimizer state rather than an implicit conversion.

Compatible CUDA FP32 Adam groups without cautious decay use PyTorch's native
fused Adam update, retaining per-parameter device counters and checkpointed
moments. Nonzero/nonfinite skip flags leave both weights and optimizer state
unchanged. CPU, cautious-decay, and incompatible layouts retain their existing
arithmetic. Gated NorMuon gradients are selected once per matrix-shape group and
reuse the packed storage for Nesterov directions, rather than launching one
selection per parameter. Fusion is numerically equivalent, not bitwise identity.

The raw structured training CLI exposes `--critic-state-read true` (default
`false`). The critic's central latent and value queries address observation
context through attention but are not added to the residual content. Each read
normalizes the attention output with a non-affine RMSNorm, then applies the
existing gated FFN. The core entrance remains non-affine normalized. Read
attention is an input projection and stays nonzero-initialized even when
`zero_init_branches` zeros residual branches; the actor and other blocks are
unchanged. Central/value query parameters initialize at unit RMS as addresses,
not constant residual shortcuts.
Actor-only BC warm starts permit differences in `critic_core_layers`,
`critic_latents`, and `critic_state_read`; actor-affecting fields must still match.
Training resume requires complete model-configuration identity. Start a fresh
critic and optimizers for this architecture. The failed `critic_unit_rms`
experiment has no active flag or compatibility alias; its checkpoints require
their original frozen source rather than reinterpretation as state-read models.

Polar Express guards a zero normalization denominator without adding a fixed
epsilon to nonzero momentum norms. Small PPO momenta therefore retain the same
normalization as larger copies, up to floating-point error. Its tensors are
float32; multiplication accuracy still follows the process-wide matmul setting.

The default physical minibatch ceiling is 6400: a complete 230080-state
production wave uses 36 balanced minibatches, with no dropped states or
gradient accumulation. The measured 6400-row update is faster than 4800 while
retaining VRAM headroom. Larger batches reduce optimizer steps per wave and
change gradient statistics; they are not learning-equivalent merely because
sample coverage is unchanged. Learning rates, objectives, and precision are
unchanged; the full learning run measures the resulting optimization dynamics.

Raw `train_ppo.py --autocull` optionally enables a single-learner online-proxy
plateau guard. Frozen-actor waves do not count. After 20 actor-active warmup
waves, either a 1000-money increase or a value-loss decrease of
`min(0.01, 1% of the reference loss)` in the alpha-0.1 EMA resets patience.
The relative cap keeps small scalar-MSE improvements visible; raw MSE and
HL-Gauss cross-entropy are not comparable strength metrics.
Thirty waves without either improvement force a
recovery checkpoint, emit `AUTOCULL`, and exit 75. State and configuration are
checkpointed; use MLQ `--max-attempts 1`. These signals are not external
strength: a collapsing policy can make value fitting easier and keep resetting
patience. External before/after games remain necessary.

Each full checkpoint binds the immutable `league/` sidecar archive with a
SHA-256 manifest, the complete source identity, and canonical calibration/run
provenance. Keep the content-addressed source snapshot and `league/` directory
beside the run artifacts. Resume through the launcher, not raw `train_ppo.py`;
the launcher restores the complete production data, league, evaluation, and
compile configuration. `--resume` supports a checkpoint outside the target run
directory, while omitting it still discovers `--run-dir/latest.pt`:

```bash
mlq submit --name kagg-ppo-resume --max-parallel-runs 1 --priority 0 \
  --time-limit 168h --cwd "$snapshot" \
  --env PYTHONPATH="$snapshot/src" --env PYTHONDONTWRITEBYTECODE=1 \
  --env CARGO_TARGET_DIR="$repo/artifacts/cargo-target/$digest" -- \
  "$repo/.venv/bin/python" scripts/launch_calibrated_training.py \
  --eager-report "$repo/artifacts/benchmarks/eager-ppo.jsonl" \
  --mixed-report "$repo/artifacts/benchmarks/mixed-ppo.jsonl" \
  --compiled-report "$repo/artifacts/benchmarks/compiled-ppo.jsonl" \
  --run-dir "$repo/runs/ppo-resumed" \
  --resume "$repo/runs/ppo-main/checkpoint-000100.pt"
```

PPO recovery checkpoints are committed only at completed update boundaries,
every 420 monotonic seconds (the accepted range is 300–600 seconds), plus
nonduplicate initial and clean-final events. Each event is serialized once as
an immutable `checkpoint-N.pt`; `latest.pt` is an atomically replaced regular
hard link to that file, so it remains directly resumable without a second full
write.

Training and benchmark metrics are mirrored to TensorBoard only after their
canonical JSONL record is durably committed. On restart, a missing, stale, torn,
or corrupted TensorBoard mirror is rebuilt from JSONL. Existing journals can be
migrated idempotently with:

```bash
.venv/bin/python scripts/jsonl_to_tensorboard.py runs/*/metrics.jsonl artifacts/benchmarks/*.jsonl
.venv/bin/tensorboard --logdir_spec runs:runs,benchmarks:artifacts/benchmarks/tensorboard
```

JSONL remains the compact, hashable calibration/provenance evidence;
TensorBoard is the primary human-facing view. `scripts/ml_pipeline_status.py
--heal --watch` prints compact pipeline state and retries bounded infrastructure
launch failures. Rerunning the calibrated launcher automatically resumes a
valid atomic `latest.pt` instead of starting over.

Screen a checkpoint on fixed, training-disjoint seeds and both seat
orientations. GPU screening and selection are throughput work, so both go
through MLQ. The default opponent is the fixed public v27 reference; any failed,
truncated, or non-finite game invalidates the result instead of being silently
excluded:

```bash
mlq submit --name kagg-checkpoint-screen --max-parallel-runs 1 --priority 0 \
  --time-limit 2h --cwd "$snapshot" \
  --env PYTHONPATH="$snapshot/src" --env PYTHONDONTWRITEBYTECODE=1 \
  --env CARGO_TARGET_DIR="$repo/artifacts/cargo-target/$digest" -- \
  "$repo/.venv/bin/python" scripts/evaluate_checkpoint.py \
  --artifact "$repo/runs/ppo-main/checkpoint-000100.pt" \
  --seed-domain screening --seeds 32 --device cuda \
  --output "$repo/evaluations/checkpoint-000100-v27-screen.json"
```

For accelerated official development/screening games, explicitly add
`--cuda-bf16-compiled --workers 1 --batch-size 32` alongside `--device cuda`.
The evaluator compiles and warms the fixed-size BF16 forward before game
clocks, pads incomplete inference waves without adding scored games, and runs
Python opponent files through independent official per-game agents. Compare
checkpoints using identical seeds, seats, batch size, and execution mode.
Reports record the warmup, precision, and backend; CUDA results do not establish
CPU submission parity. Default CPU admission behavior is unchanged.

The 32-seed screening panel ranks candidates; it is not final admission evidence.
Freeze the selected checkpoint before running the untouched finalist panel.
Both seats and all opponents on one map form one independent seed cluster.
Score intervals use bounded Hoeffding uncertainty, including unanimous outcomes;
selection reports also retain paired candidate differences. The bounds are
conservative and do not turn adaptive screening into held-out evidence.

Reserved map domains are BC `[0,4000000)`, development `[4000000,8000000)`,
screening `[10000000,11000000)`, finalist `[12000000,13000000)`, and online RL
`[20000000,2**32)`. Artifacts bind actual BC train/holdout seeds and planned
training/development exposure. Explicit seed overrides must stay in their domain.
Reports validate recorded exposure; maintaining untouched finalist maps across
separate invocations remains a procedural requirement, not a global ledger.

To screen every numbered checkpoint on identical paired seeds and atomically
promote the strongest lower-confidence-bound result:

```bash
mlq submit --name kagg-checkpoint-select --max-parallel-runs 1 --priority 0 \
  --time-limit 8h --cwd "$snapshot" \
  --env PYTHONPATH="$snapshot/src" --env PYTHONDONTWRITEBYTECODE=1 \
  --env CARGO_TARGET_DIR="$repo/artifacts/cargo-target/$digest" -- \
  "$repo/.venv/bin/python" scripts/select_checkpoint.py \
  --run-dir "$repo/runs/ppo-main" --seeds 32 --device cuda \
  --output "$repo/evaluations/ppo-main-screen.json" \
  --best-output "$repo/runs/ppo-main/best.pt"
```

Package admission uses the CPU execution contract matching Kaggle. Queue this
work too. The public finalist report requires the selection report and its frozen
artifact identity; a standalone evaluation cannot bypass selection provenance.
Then run the mandatory built-in `starter` gate:

```bash
mlq submit --name kagg-finalist --max-parallel-runs 1 --time-limit 2h \
  --cwd "$snapshot" --env PYTHONPATH="$snapshot/src" -- \
  "$repo/.venv/bin/python" scripts/evaluate_checkpoint.py \
  --artifact "$repo/runs/ppo-main/best.pt" \
  --selection-report "$repo/evaluations/ppo-main-screen.json" \
  --seed-domain finalist --seeds 32 --device cpu \
  --output "$repo/evaluations/ppo-main-finalist-v27.json"

mlq submit --name kagg-starter-admission --max-parallel-runs 1 --time-limit 2h \
  --cwd "$snapshot" --env PYTHONPATH="$snapshot/src" -- \
  "$repo/.venv/bin/python" scripts/evaluate_checkpoint.py \
  --artifact "$repo/runs/ppo-main/best.pt" \
  --opponent starter --seed-domain finalist --seeds 16 --device cpu \
  --selection-report "$repo/evaluations/ppo-main-screen.json" \
  --output "$repo/evaluations/ppo-main-starter.json"

PYTHONPATH="$snapshot/src" "$repo/.venv/bin/python" \
  "$snapshot/scripts/build_submission.py" \
  --checkpoint "$repo/runs/ppo-main/best.pt" \
  --evaluation-report "$repo/evaluations/ppo-main-finalist-v27.json" \
  --builtin-evaluation-report "$repo/evaluations/ppo-main-starter.json" \
  --output "$repo/artifacts/kaggriculture-ppo.tar.gz"

mlq submit --name kagg-bundle-validation --max-parallel-runs 1 --time-limit 2h \
  --cwd "$snapshot" --env PYTHONPATH="$snapshot/src" -- \
  "$repo/.venv/bin/python" scripts/validate_submission.py \
  --archive "$repo/artifacts/kaggriculture-ppo.tar.gz" \
  --opponent v27 --seeds 2 \
  --output "$repo/evaluations/kaggriculture-ppo-bundle.json"
```

## Game mechanics

The [mechanics overview](mechanics/overview.md) documents the rules implemented by
the pinned `kaggle-environments==1.32.6` release. The companion pages cover:

- [agent API and state](mechanics/api.md)
- [default constants](mechanics/constants.md)
- [farm and shed](mechanics/farm.md)
- [farmers and farm hands](mechanics/farmers.md)
- [crops](mechanics/crops.md)
- [animals](mechanics/animals.md)
- [market](mechanics/market.md)
- [town demand](mechanics/town.md)
