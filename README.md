# kraggiculture

Research and evaluation tooling for the Kaggriculture simulation competition.

Production training is BC-initialized self-play PPO on the `lejepa` family with
DAPO's asymmetric clip band, terminal win/loss/draw rewards, and Monte Carlo
credit for both actor and critic (GAE lambda 1 on both, still separate knobs). A fresh
production run must load one behavior-cloned actor, then fits its fresh critic
for at least ten iterations and until every member's previous fresh-wave
pre-update Monte Carlo-return R-squared reaches 0.10. Only an existing
checkpoint can bypass that initialization. An exact batched Rust simulator supplies
high-throughput rollouts; the pinned Kaggle environment remains the parity
oracle and final evaluator.

The production `lejepa` critic is the exact win/draw/loss classifier
(`--wdl-value`, `LejepaConfig.wdl_value`); every other family's default, and
`lejepa` with `--wdl-value false`, uses categorical HL-Gauss cross-entropy with 255 linearly
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

The production model family is **`lejepa`** ([LeJEPA world model](#lejepa-world-model)):
one `entity-attention` trunk, described next, shared by actor and critic and trained
by its world-model objective beside the policy's loss. Its defaults are the promoted
core -- schema v4, every market quantity choice (`--action-interface 2`), the local unit
affordance scorer, unit target navigation, market resource conditioning, the critic
readout FFN and the WDL critic -- which the adopted PPO recipe was measured on.

The `entity-attention` trunk has width **96** for both
encoded memory and entities. Two shared-weight local transformer blocks encode
each 100-tile farm independently. Exactly **26 persistent states**—16 unit slots
and ten market-order slots—perform four rounds of self-attention, static-memory
cross-attention, and one FFN by default. Four query heads share two KV heads.
Default memory normalization and K/V projection happen once per forward and remain
differentiable through every round. Economy-conditioned RMS scale/shift modulates
each branch.

Actor memory contains all **200 farm tiles plus 20 economy tokens**. Unit queries
retain HERE/N/S/E/W local features; tile coordinates and ownership remain explicit.
Inactive units are masked as attention keys and zeroed after every branch;
all ten market slots remain valid before sampling. Final normalized unit/market
states feed the existing policy heads directly by default, without generic latents,
scratch tokens, opponent summaries, reinjection, MUDD, or separate output decoders.

The independent centralized critic adds **16 private opponent-unit memory
tokens** (236 total), but still evolves only 26 entities. One learned critic query
uses projected **four-query-head/two-KV-head cross-attention** over the valid entity
states and all valid source-memory tokens by default, bypassing the entity-only
value bottleneck. Its normalized `[B, 1, 96]` output feeds the HL-Gauss head and the existing
critic NextLat target. The default readout has no extra FFN, query residual shortcut,
or persistent core token. Residual zero-initialization does not zero this standalone readout.
In the `entity-attention` family actor and critic share no parameters. Every attention
layer in that family, including critic NextLat, uses GQA, except the explicit BiXT cross-attention
ablation below; configurations with equal query/KV head counts are rejected.

Selectable entity architecture flags are available. Production
defaults to untied projected memory; the table lists the available alternatives
and ablations:

| Ablation | CLI override | Effect |
| --- | --- | --- |
| Shared projected memory | `--shared-memory-kv true` | One K/V projection and key normalization are shared across rounds; common source normalization still runs once. |
| Intermediate FFN | `--inter-attention-ffn true` | Both networks use SA → FFN → CA → FFN, with independent gated, economy-conditioned branches. |
| Final local unit readout | `--unit-local-readout true` | After global reasoning, each actor unit reads its five HERE/N/S/E/W tiles through a local attention/FFN block. Market states and the critic do not gain this decoder. |
| Critic readout FFN | `--critic-readout-ffn true` | A gated pre-RMS FFN follows learned-query pooling, before final normalization and HL-Gauss. Actor artifact identity is unchanged. |
| Remove local initialization | `--unit-local-init false` | Omit the five-tile initialization projection and the private opponent-unit HERE relation tag, retaining categorical, position, ownership, and continuous unit information. |
| Tile cross-attention RoPE | `--tile-cross-rope true` | Rotate unit queries and both farms' tile keys in global entity-memory reads using the existing axial 2D RoPE; no added parameters. |
| Entity-only critic readout | `--critic-source-read false` | Disable the default source-memory read and pool only over entity states. Actor artifact identity is unchanged. |
| Economic valuation critic | `--critic-architecture economic` | Replace the critic's decision-slot trunk with 24 valuation states reading 252 centralized sources. Actor artifact identity is unchanged. |
| Midpoint memory writeback | `--memory-writeback true` | After half the entity rounds, source tokens read entity states through an attention/FFN block; later rounds use fresh memory projections. |
| Own-unit tile bias | `--unit-tile-bias true` | Add a learned, initially zero per-head displacement bias only between own-unit queries and own-farm tile keys. Mutually exclusive with tile cross-attention RoPE. |
| BiXT core | `--bixt-latents 32 --global-modulation false --critic-source-read false` | Refine generic learned latents and data tokens using one shared similarity matrix with separate row/column softmaxes. Read action/value heads from refined decision tokens. |

The local readout uses already encoded farm tiles, not a second encoder. When
initialization and final readout are both enabled, they share one local gather.
Invalid neighbors cannot affect the readout; inactive unit outputs remain zero.
The local-initialization experiment compares readout-on/init-off against
readout-on/init-on, rather than changing both features relative to the default.
For BC ablations select `--architecture entity-attention`; `--production-model`
intentionally locks the production recipe. Actor-changing flags require matching
BC artifacts; critic-only changes can reuse the default actor. Critic source read
and hardness league selection are production defaults. The historical architecture
campaign explicitly retains entity-only critic pooling and stratified selection
in its control, enabling each promoted feature only in its named arm.

The opt-in economic critic has independent farm, unit, and private-economy
encoders. Its **252 source tokens** are 200 tiles, 20 economy tokens, 16 own-unit
slots, and 16 private opponent-unit slots. **24 valuation states** start from
two farm summaries, two masked workforce summaries, and the 20 economy tokens;
GQA rounds exchange information between these states and read the full sources.
Inactive units are masked, including when a workforce is empty. Final pooling
retains the `[B, 1, 96]` belief, HL-Gauss head, and existing critic NextLat objective.
This tests a dedicated value representation; it adds no economic forecasting
targets. It shares no actor parameters and has no actor market-order queries.
`--critic-source-read` applies only to the default `entity` critic's final source
bypass; economic valuation states always read all sources. Local unit initialization,
local readout, tile cross-RoPE/bias, and memory-writeback flags retain their actor
effect but do not change the economic critic. Economic and BiXT are separate,
incompatible experiments; `--critic-architecture entity` remains the default.

Three larger, opt-in structural ablations preserve GAE and carry no recurrent
state across turns. `--architecture strategic-plan` uses a global workspace and
one categorical plan shared by every action head; BC marginalizes plans exactly
and PPO records the sampled plan. `--architecture causal-execution` conditions
36 action factors on the selected prefix and an exact native-compatible resource
ledger, using cached on-device generation and parallel teacher forcing.
`--critic-architecture forecast` adds action-conditioned economic predictions at
1, 24, 96 and 384 steps while the value head remains state-only. Its control is
the same economic trunk without forecasting heads.

`--architecture lejepa`, the production family and the default architecture of
`train_ppo.py`, is a single `entity-attention` trunk shared by the actor and the
critic, trained by an attached-target world-model objective and the policy's loss,
and read through private rounds of cross-attention by both the actor's decision
slots and the critic's tower. It is described in full under
[LeJEPA world model](#lejepa-world-model) below.

`scripts/queue_structural_campaign.py` freezes full-budget comparisons with
matched BC data, separate numerical/production gates, both decoding modes, and
fixed-panel culling. The [architecture design](ARCHITECTURE_ABLATIONS_20260918.md)
and [cross-run baseline comparison](RUN_COMPARISON_20260918.md) explain the
controls and current default choice. These experimental architectures are not
production defaults.

`scripts/queue_credit_campaign.py --baseline PATH --submit` freezes and queues
the historical full-production comparisons of actor GAE lambda 1, critic NextLat
off, and the economic critic. These named experiments explicitly retain NextLat
on in their controls despite the new production default. The baseline is a
submitted combined-default campaign manifest. Each candidate passes compiled
CUDA contracts and a production benchmark before training. This historical
builder gates evaluations on successful training; valid interrupted or culled
checkpoints require a separate evaluation, as recorded in the run comparison.
`scripts/summarize_credit_campaign.py CAMPAIGN` checks source
and checkpoint identities, reports failures separately from learning evidence,
and pairs outcomes against the same BC actor and baseline. A preview without
`--submit` reserves its campaign name, so use a distinct `--name` for previews.

The BiXT arm follows the shared-reference, simultaneous bidirectional update from
[the paper](https://arxiv.org/pdf/2402.12138v2), with separate feed-forward updates
on both streams followed by latent self-attention. It keeps the farm encoders,
width, depth, policy/value heads, RMS normalization, ReLU-squared FFNs and gated
residuals; bidirectional attention uses full MHA while latent self-attention uses
GQA. Shared references use input RMS normalization, without the control's
additional per-head Q/K normalization; latent self-attention retains it.
FiLM is disabled explicitly. This is an adapted architecture comparison,
not an exact ImageNet-model reproduction or a single-factor attention ablation.
The actor's data sequence has 246 tokens (26 decisions plus 220 source tokens),
and the critic has 262. At depth four, the first two rounds update all data;
the third still reads all data into latents but only writes decision tokens;
the fourth only reads latents into decisions. Omitted writes cannot reach either
head, and the final round contains no unused latent-update parameters.

Source-read, writeback and BiXT training use granular activation recomputation
to preserve the full production minibatch. No-grad collection remains direct.
Stateful fused MLPs are excluded from recomputation; the BiXT arm requires them
disabled. The default critic uses recomputation for its source read; the default
actor does not.
BiXT training additionally bounds attention scratch inside opaque CUDA operators:
independent batch rows are tiled internally, without changing the PPO minibatch
or optimizer update. Both directions still share each score matrix. Backward
recomputes probabilities instead of retaining them across the token FFN; no-grad
rollout keeps the dense compiled primitive. This is not a fused attention kernel;
production memory and throughput are measured in the experiment gates.

The default `--league-selection hardness` pools the configured lane budgets
and selects distinct opponents with the lowest recent posterior learner score.
One lane refreshes stale/uncertain evidence. When unseen opponents remain, a
separate discovery lane first admits untested built-ins, then the newest untested
snapshot. This prevents a large stale archive from starving new strategies;
older unseen snapshots remain eligible for refresh. With the production twelve
lanes (two active, six historical, four built-in), ten remain for known hardness
while discovery is needed, otherwise eleven. One- and
two-lane budgets rotate exploration purposes deterministically across waves.
Beta(1,1) shrinkage handles sparse evidence, exact ties are randomized, and
effective game counts decay by 0.98 per learner wave. Games remain evenly
distributed across selected opponents with balanced seats. Evidence comes only
from current-learner games, uses native terminal outcomes, and persists in
recovery checkpoints. `--league-builtin-lanes 0` disables built-ins.
`--league-selection stratified` retains the earlier age-stratified PFSP ablation.

Cross-attention RoPE uses unit `(column, row)` coordinates and each farm's local
board grid; farm ownership remains in the encoded features, not an invented
coordinate offset. Market queries, economy keys, private opponent-unit keys, and
all values remain unrotated. Mixed spatial/nonspatial scores still change because
one side is rotated. Shared tile keys are rotated once before the entity rounds;
untied keys are rotated after each round's projection and key normalization.
The flag does not change entity self-attention, critic pooling, or the optional
five-tile final decoder. Farm self-attention already uses 2D RoPE by default.

Ordinary CUDA GQA folds query-head groups into the query sequence without repeating K/V.
BF16 attention uses FlexAttention's standard Triton kernel for key-masked GQA
updates, fusing validity into dense scores without dynamic sparse-block metadata.
Unmasked calls prefer Flash; masked no-grad rollout prefers cuDNN, retaining its
eight-opponent `vmap` batching (unsupported by this installed FlexAttention).
The efficient CUDA kernel remains available for unsupported SDPA configurations.
Short folded attention (head width up to32, up to64 queries and256 keys) uses
64×64 Flex tiles; larger geometries retain backend heuristics. This avoids
overpadding the entity queries with the default128-row backward tiles.
Masked gradient attention above65535 batch rows uses32768-row attention views to
stay within CUDA's grid-Y limit; this covers the8192×16 local unit decoder without
reducing the PPO minibatch or switching precision/backend.
Folding avoids native Flash GQA's expanded backward K/V intermediates. Masks remain
part of the model: inactive entity states are zeroed after each reasoning branch,
and excluded as attention keys. Removing masks is an architecture change, not an
equivalent dense execution strategy.

`inductor_graph` rollout already captures active-learner and batched frozen-policy
inference together; native stepping and sampling remain outside that graph.
For `reduce-overhead` or `max-autotune` updates, one explicit CUDA-graph iteration
spans the complete PPO minibatch: actor gradients remain valid through critic
backward and the guarded optimizer steps. Replay-only chunks copy their outputs
before advancing the graph boundary. Graph modes are supported, not presumed
faster; compare complete warm iterations on the actual architecture and schedule.

Existing `entity-attention` BC actors remain compatible with this critic change.
Old scalar-pool critic checkpoints are intentionally incompatible with the new
readout: use their original frozen source to resume them, or initialize a fresh
critic. Historical `structured` and `conv-entity` artifacts retain their own
registered architectures; their weights are not compatible initialization for
the entity family. Train an entity BC artifact with `--production-model` when
migrating from those families.

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
`kaggle-environments==1.32.7` before they are admitted to training.

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

The production default is an HL-Gauss source-read critic with **actor and critic
NextLat disabled**, using ordinary value fitting and individual-state PPO
shuffling. The [cross-run comparison](RUN_COMPARISON_20260918.md) records this
decision. Critic NextLat is opt-in with
`--structured-critic-latent-coefficient 1 --structured-critic-value-coefficient 1`:
this adds latent SmoothL1 and decoded-value loss at horizon 1. Source gradients
are not norm-matched. For a scalar critic, decoded-value KL is unit-variance Gaussian KL
(half squared mean error), not a degenerate one-category softmax. Distributional
critics use categorical decoded KL over the same capped logits as their readout.

Actor and critic use separate backbones and entity states, with no shared
parameters. The value loss remains attached through the critic's attention pool
to its entity rounds and tile encoder; it never updates the actor backbone. Critic
NextLat also trains its source backbone and predictor, but detaches its
successor teacher and the value-head weights used for auxiliary decoding.
Its categorical KL is teacher-to-student over the full value distribution.

Actor NextLat is opt-in: set `--structured-latent-coefficient 1` and
`--structured-decision-coefficient 1`, with `--structured-decision-horizon 1`
for one-step prediction. `ActorDynamics` has independent unit-action, market-kind,
and market-quantity residual MLP predictors. Each follows the NextLat reference:
RMS-normalize the concatenated action/state, apply three bias-free linear layers
with two GELUs, then add the predicted delta to the source state. At D96 the
hidden width is 256. A shared, fixed-slot projection of the valid joint action
conditions all three predictors; inactive units, post-STOP slots and quantities
on non-quantified kinds cannot affect that action code.

The sources are normalized head-input representations. In `entity-attention`
these are final entity states; legacy `structured` uses decoder outputs. Kind
and quantity share the source market state but evolve independently. Each head's
SmoothL1 is independently normalized over eligible successor coordinates, and
the three means are **summed**. Unit targets require cumulative survival;
newborn successor units are excluded. Kind targets use reached successor orders,
while quantity targets require successor quantity activity.

Decoded teacher-to-student KL uses only the **frozen final policy projections**,
not a replay of the entity decoder. Cached, detached successor head-input
representations supply the teacher. Student and teacher projections use identical
FP32 arithmetic with successor legality and activity masks. Quantity decoding
freezes its complete D96-to-rank32 factorized readout and uses the successor
selected kind. The three independently normalized head KL means are **summed**.
Thus coefficients 1/1 apply one latent and decoded objective per head, without
an implicit division by three, extra cross-entropy, or gradient balancing.

PPO and the auxiliary share one actor forward and one additive combined backward.
Auxiliary gradients enter the live source head-input representations and flow
through the actor trunk; successor targets and auxiliary
readout weights are detached. Normal PPO gradients still train the final policy
projections. There is no actor source-gradient balancing.

The auxiliary-enabled actor trunk uses non-reentrant activation checkpointing:
backward recomputes its intermediates without splitting the PPO minibatch or
adding optimizer steps/backward calls.

This configuration remains experimental: contract tests establish gradient and
execution correctness, not improved learning. Actor/BC parameter keys are unchanged
within each registered family, but weights and predictor states from different
architectures are not interchangeable. Warmup release still measures critic
readiness, not predictor readiness; persistence scores remain diagnostic only.
Old shared-attention actor-predictor states are not compatible with these three
MLPs; start fresh predictor state rather than silently migrating a resumed run.

### LeJEPA world model

`--architecture lejepa` replaces the detached NextLat convention with LeWorldModel
(arXiv:2603.19312), which is LeJEPA (arXiv:2511.08544) applied to transitions.

There is **one encoder in the family, trained by this objective and by the
policy's loss.** The actor owns it as `trunk.`, so snapshots, bundles and frozen
ensembles are unchanged; the critic borrows the same module rather than holding
one of its own and reads its latents through a detach, so no value gradient
reaches it. The actor's heads read the attached belief
(`LejepaConfig.policy_shapes_backbone`, on by default), so the clone loss and
then PPO's policy loss arrive in the same backward as the objective's gradient.
Anti-collapse is SIGReg's job, not an asymmetry in the graph, so a second copy
would buy nothing but a second copy's compute. The objective's optimizer is the
backbone's optimizer either way -- it clips the encoder's summed gradient and
steps it beside the projector and the predictor, because they are one model --
and `update_ppo` refuses to run if anything else claims those weights.

The policy gradient is not optional. With it detached
(`policy_shapes_backbone=False`, kept as the ablation) a 12-epoch clone matched
the teacher at argmax (0.997 market-kind accuracy) and went bankrupt in every
temperature-1 self-play game: its farm went unwatered and its animals unfed
within two days of the first sampled departure, because features fitted only to
predict the teacher's trajectory carry nothing a decision needs once play leaves
it. The attached clone (holdout NLL 0.0005 vs 0.0033) plays teacher-level banks
of 70-126k sampled, and its objective is healthier, not weaker: motion 0.44 vs
0.17 at the same dispersion.

The encoder steps **exactly when the policy does**, on the same minibatch and at
the rate `--structured-learning-rate` sets for the world model. A `lejepa` launch
defaults it to 1.5e-5 (`LEJEPA_PPO_DEFAULTS` in `ppo.py`), a tenth of the former
1.5e-4 actor rate: at the full rate the encoder moves under both gradients every
minibatch, and the slower backbone beat the full-rate run on the same clone
(JEPA_RUNS, 9281 vs 9275). The objective's weights default the same way, to
prediction 1.0 and SIGReg 0.09 with the reward head off; every other family
defaults to no objective and to the actor's rate. The heads read it, so every encoder step is a policy step
whether or not the actor's optimizer took one: a
minibatch after a KL stop that moved the encoder would move the policy past the
trust region that had just refused to. After a KL stop, through the critic's
extra epochs and through a warmup wave, the encoder is frozen with the policy and
only the projector, the predictor and the reward head keep fitting against it.
The encoder's gradient is dropped rather than zeroed there, because the optimizer
skips a missing gradient outright while a zero one would still move the weights
by momentum, and the backbone's parameter groups keep a warmup clock of their
own that counts only the steps it takes -- a warmup wave spent fitting a fresh
predictor must not use up the backbone's warmup, or its first policy steps would
run at full rate beside heads still warming up. All three losses are computed on
the pre-step parameters and the
three optimizers then step together, so the world model, the actor and the
critic see the same rows the same number of times, in an order that makes no
tower train against another's already-moved weights.

A warm start (`--init-actor-from`) trains the encoder before PPO by behavior
cloning (below), so its warmup waves freeze a trained backbone rather than a
random one: they fit the critic, and fit the objective -- above all the reward
head, which a clone never trains -- against the cloned encoder. Release is gated
on the critic's fit alone, not on the objective's, so watch the
`structured_preupdate_*` ratios over the first released waves. A run started
from scratch has no warmup and releases a random backbone and its objective
together. The critic reuses the actor's encoding of each minibatch rather than
running the backbone a second time.

The objective has exactly two terms plus one this environment forces:

* **Attached next-step prediction.** The regression target is the encoder's own
  embedding of `o_{t+1}`, with its gradient live. Every other self-predictive
  objective here detaches that target; this one does not, because the anti-collapse
  job belongs to the distributional constraint below rather than to an asymmetry in
  the graph.
* **SIGReg.** Random unit directions are drawn through each latent group and every
  one-dimensional marginal is pushed onto `N(0, 1)` by a sliced Epps-Pulley
  characteristic-function statistic (17 knots to `t = 3`, Gaussian window). A
  collapsed embedding has a degenerate marginal in every direction and is maximally
  penalized. The reference's `* n` scaling is dropped, so the coefficient is
  invariant to minibatch size, token count and unit occupancy; token positions are
  then combined in proportion to their own support, which is that same factor
  restored per position and divided out once at the end.
* **Reward.** The one part of the next step an encoded observation cannot contain.
  Its head sits outside the transition round and is scored on every staged row,
  because under `terminal-outcome` rewards the whole nonzero signal sits on each
  episode's final step, which is never a transition source.

Nothing is corrupted: observations enter the encoder exactly as the policy sees
them, and the only supervision signal is the passage of one step. There is no
temporal conditioning either -- the predictor reads the current step's embedding
and the executed joint action and nothing else, which is precisely what the policy
and the critic on this encoder are allowed to know.

The latents are per token rather than one pooled CLS embedding, and SIGReg runs
per group (`unit`, `market`, `economy`, `tile`) *and per token position within a
group* -- the reference's
reduction, which averages over the batch axis alone. Both splits are load-bearing,
because a normality test over a union is satisfiable by a mixture that is
degenerate inside every component: a farm tile and a market-order slot share
neither support nor scale, and an encoder that hands every tile slot its own
constant has a prediction loss of exactly zero while its pooled cloud looks
entirely ordinary. Per position that encoder scores worse than total collapse.
Positions no row supervises are dropped rather than scored against an empty
sample, and inactive unit slots -- which carry an exactly zero latent -- are
excluded from every population. Support weighting is what makes that masking safe:
a position's statistic falls like `1 / n_position`, so weighting positions equally
would let one thinly supported slot -- a unit the agent has just learned to hire --
read as collapse on sampling noise alone, and pull that slot toward the origin.
Under a uniform mask it is identical to the plain mean.

Cost is bounded three ways. Contiguous-run minibatches already place a row and its
successor in the same batch, so the objective costs **no additional encoder
forward**. Thirty-two farm tiles are supervised per minibatch out of two hundred
encoded, resampled every minibatch from the trainer's auxiliary generator -- a
budget on what is supervised, never a corruption of what is encoded. In eager the
Epps-Pulley intermediate is accumulated in fixed-size checkpointed chunks; under
`torch.compile` that loop is skipped, because Inductor's partitioner already
decides what to keep across it and a Python-level chunk loop would only multiply
the traced subgraphs. Slice directions and the tile sample are drawn on the host
and copied in place into non-persistent buffers whose identity, shape and dtype
never change, so the compiled update graph traces once and a refresh forces no
recompilation. The reward term is skipped outright at a zero coefficient (its baseline
`reward_scale` is zeroed with it, so "not scored" is not read as "scored
perfectly"), and its head pools each latent group separately rather than
concatenating them.

Enable it with `--architecture lejepa --jepa-prediction-coefficient 1
--jepa-sigreg-coefficient 0.09 --jepa-reward-coefficient 0.1`. Prediction and
SIGReg must be positive together: an attached target alone is minimized exactly by
a constant encoder, and a collapsing run's loss curve is a clean descent to zero.
The detached NextLat coefficients are refused. There is one arm, so there are no
`--jepa-critic-*` flags and no critic predictor: a `lejepa` critic admits no
dynamics predictor at all, and a `lejepa` actor admits only this objective, its
belief class being keyed on the family so that a detached NextLat would walk a
four-field belief into a helper written for one.

The actor reads the belief the same way, through `policy_readout_layers` rounds
(default 1) in which its unit and market slots cross-attend to the whole
observation before their heads. Per-slot heads can only use what the world
model chose to keep in each slot, and it has no reason to copy the economy into
a market slot when the economy tokens already hold it: a 12-epoch clone through
a gated feed-forward per slot plateaued at 0.76 market-kind accuracy, against
0.997 with the readout and the entity actor's 0.999. `--policy-readout-layers 0`
keeps the linear probe of each slot as a control.

The critic's privileged information arrives strictly *downstream* of the shared
encoder, which is what makes sharing safe. The backbone encodes only this seat's
own observation -- the critic slices the private economy columns off before
calling it -- and the opponent's unit slots are embedded by the critic's own
parameters and folded into its detached latents by `critic_private_layers` rounds
of cross-attention. Nothing privileged can reach the actor through weights
the actor evaluates, and nothing privileged is in the population SIGReg scores.
The opponent's units stay out of the predicted belief for the same reason as
before: a Markov predictor conditioned on this seat's action could only learn
their mean.

Behavior cloning trains this family with its objective beside it: the clone
loss reaches the encoder through the heads, and the objective is what makes the
encoder a world model rather than an entity trunk under another name (and, under
the detached ablation, the only thing that trains it at all).
`scripts/train_bc.py` therefore requires the LeJEPA objective whenever it clones `lejepa` (and refuses it for any other family):
`--jepa-prediction-coefficient` and `--jepa-sigreg-coefficient` must both be
positive, `--jepa-horizon` sets the transition horizon, and `--run-length` must
be at least `horizon + 1` so each minibatch holds its successors. The encoder is
then trained on the demonstration transitions by prediction and SIGReg while the
heads clone; there is no reward in a demonstration, so the reward term and its
baseline are off and read as zero in the journal. It is trained as PPO trains
it: runs are re-cut every epoch at a phase drawn per episode, so every
transition is a source rather than the half a fixed cut would pick, and the
heads and the world model are gradient-clipped apart. The objective travels in the
artifact beside the actor (`jepa_objective`), and PPO's `--init-actor-from`
resumes it with the encoder and heads: the projector and the predictor are one
model with the backbone, and fresh ones would spend the critic warmup relearning
what the clone already fitted, then pull the released backbone toward whatever
embedding they had settled on. A `lejepa` warm start refuses an artifact
without one -- including when an in-flight run is resumed with its original
`--init-actor-from`, if that clone predates the key: re-clone it. A `lejepa`
training checkpoint written before the backbone had its own optimizer groups
does not resume either. The warmup waves still fit the objective against the frozen
backbone, which is where its reward head first sees a reward. The structural
campaign's `lejepa` arm does exactly this: BC at `--run-length 2` with prediction
and SIGReg, then the gate and PPO with the reward term added.

What this does not defend against, stated plainly because the telemetry measures
it: a one-step attached target is also minimized by an encoder constant *along a
trajectory* while varying across the batch, and a minibatch of consecutive pairs
carries almost no within-episode structure for a normality test to reject. The
defenses are the reward term and the two journaled controls
(`structured_preupdate_persistence_*`, `structured_preupdate_shuffled_*`) -- a
prediction loss that matches persistence, or that survives shuffling actions
across rows, is measuring nothing. The shuffled control rolls the action tokens
over the transition sources by the run length rather than by one: each
contiguous same-trajectory run contributes `run_length - horizon` consecutive
sources, so a roll shorter than that would mostly hand a source its own
trajectory's neighbouring action, which is not a control at all, while a roll of
the run length always lands in a different run. But persistence alone
cannot separate the two, because a trajectory-constant encoder sends that baseline
to zero along with the loss, and `structured_actor_dispersion` stays at one
throughout: across the batch such an embedding is still perfectly normal.
**`structured_actor_motion` is the column to watch and to cull on by hand**
(nothing gates a run on it automatically) -- the RMS
displacement of the *latents* the policy and the value function read (not the
projected embedding, which is discarded after training) between a row and its
successor, relative to their own scale and over supervised tokens only. That
number going to zero *is* the failure.

Both controls are journaled as ratios against the live loss under
`structured_persistence_*` / `structured_shuffled_*`, over `combined`,
`prediction` and each latent group. Only the prediction
columns are compared: the controls share the live objective's projector, SIGReg
and reward head, so every other column is identical by construction.

`JepaObjective` is training-only. League snapshots, inference bundles and frozen
ensembles consume the actor's state dict whole and never carry a projector or a
predictor the deployed model does not evaluate.

Both coefficients default to zero, so no predictor and no predictor optimizer are
constructed -- and on this family that also means the backbone never steps, which
`update_ppo` treats as the configuration error it is. `--structured-critic-gradient-balance` remains an
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
This experiment requires both GAE lambdas to be one, the default, and component
ratio scope. It cannot be combined with a shorter actor trace or joint ratios.

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

The explicit `--reward-mode shaped` alternative uses discount-correct, exactly
zero-sum potential shaping. Let `L[i,t]`
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

The temporal defaults are `--gamma 1`, `--actor-gae-lambda 1`, and
`--critic-gae-lambda 1`. The critic fits full, undiscounted Monte Carlo returns:
the default target at every valid state is the terminal win/loss/draw outcome.
With explicit shaped reward, the target is instead `U[T] - P[t]`. The actor's
advantage is the same return less the critic's baseline. Collection shaping,
advantages, and value targets share gamma. HL-Gauss targets outside categorical
support saturate at the outer atom, with the saturated fraction reported.

The actor lambda was VAPO's `1 - 1 / (0.05 * 719)` = 0.972 until 2026-09-27: a
trace with geometric weight sum near 36 transitions, trading variance for the
critic's bias. Fine-tuning the eight-epoch WDL clone, whose critic reaches only
0.10-0.23 Monte Carlo R-squared, every 0.972 arm collapsed after the clone (zero
argmax wins against V27 at wave 50) while lambda 1 beat the clone itself 0.992
argmax at the same wave (`artifacts/probes/ppo-ablations-20260927`). Pass
`--actor-gae-lambda 0.972183588317107` to reproduce the former trace.

The temporal settings were first promoted from dense-reward trial **7010**.
The user subsequently selected **7122**, the HL-Gauss VAPO terminal-outcome LR3
trial, as the new production default: terminal win/loss/draw reward and tripled
actor/critic rates, retaining HL-Gauss, component PPO clipping/KL,
and the existing auxiliary recipe. The later entity-attention architecture
promotion is separate. This adopted VAPO's temporal settings, not
every component of its training recipe. New launches inherit the new defaults;
explicit overrides and previously frozen commands retain their declared settings.

The current recipe is the September 27 staged PPO ablations' adopted arm
(`artifacts/probes/ppo-stage2-20260927`, `lambda-1-actor-lr-5e-5`; stage three
confirmed keeping the JEPA objective): `lejepa` from a BC clone, actor lambda 1,
actor LR 5e-5, backbone LR 1.5e-5, minibatch 4096 with retained update
activations, 128 self-play plus 64 league games with four built-in lanes, and the
architecture panel every 25 actor-active waves. A plain `train_ppo.py` launch that
states only its run directory, clone, budget and seed resolves to exactly that
command, and `build_training_command` emits it in full.
`scripts/queue_core_campaign.py` still states every flag, because it launches
frozen source snapshots whose defaults predate these.

The default trust region is `target_kl = 0.03` on the active-component mean KL.
Its historical calibration does not establish the stopping frequency after
changing rewards and auxiliary balance; measure accepted minibatches explicitly.

Entropy is measured but not optimized. The main actor objective is clipped PPO;
production leaves the actor future-policy auxiliary off unless explicitly enabled.
The production critic has no auxiliary objective: critic NextLat is off by
default, and the `lejepa` world-model objective is mutually exclusive with the
detached NextLat terms. Production uses one learner with 128 self-play games and 64 league games per wave (320
learner trajectories). Stale matchup evidence for built-ins and snapshots decays
toward 0.5 alike, so formerly easy opponents can become contested again.

With `inductor_graph`, training precompiles balanced league layouts up to the
configured lane budget on its first nonempty league wave. This moves their cold
compilation to startup without changing assignments or padding steady waves.
Previously unseen unbalanced layout combinations can still compile later; inspect
the recorded compile events rather than treating every idle interval as GPU work.
CUDA graphs remain wave-owned; new update-gradient phases can also compile
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

Schema v4 appends `money_margin` to each farm token: that farm's signed
`log1p` money minus the other farm's, unscaled and rounded once from float64.
The absolute `money` feature (`/12`) is staged in fp16 and cast to bf16 under
autocast, which leaves roughly 5% resolution on a late-game bank; the margin is
near zero exactly when a game is close, where floating point is finest.

Schema v5 appends `shop_<NAME>_first_unlock` to the town token for each shop:
one plus the index of that shop's first instance in the town's unlock order,
over eight, or zero while it is locked. The per-shop counts are a multiset and
cannot say which shop opened first; with them, the ranks recover the order of
the distinct shops, which opening-keyed demand plans read (the `demand-advance4`
bot routes on its first two unlocked shops, and most of its routes differ from
the swapped pair's).

Schema v6 prices held stock. Each product token appends `held_value`: the exact
coins selling every unit of it the seat holds (shed and hands) would bank, one
unit at a time down the price curve as each sale restocks the market, over a full
shed at the top base price (25,000). Each farm token appends `liquidation`, money
plus those proceeds on the `money` scale -- the engine's liquidation value and
the shaping potential's input -- and `liquidation_margin`, its float64 signed
`log1p` ratio to the other row's. Held stock is private, so the opponent row's
liquidation is its money alone; the centralized critic reads the opponent's own
`held_value` as the private `opponent_held_value` column.

v3 through v6 coexist: both tokenizers always emit the v6 layout, and each
model's `observation_schema_version` selects the product-, farm- and town-token
prefixes its embedder reads (and a critic's private product prefix), so v3, v4
and v5 artifacts load and act unchanged. Fresh LeJEPA
model configs, production's, default to v4; `entity-attention` and the other
structured families continue to default to v3. Override a fresh run with
`--observation-schema-version` when making an explicit schema comparison.

Python action helpers and inference use the default shed capacity of100.
`CheckpointAgent.__call__(observation, configuration)` and the generated submission
entrypoint reject a supplied nondefault `shedCapacity` before inference.
Observation-only calls and `act_many` assume default-capacity environments;
capacity cannot be inferred from the observation.

`all_unit_action_masks` reports independent masks for the same observation
snapshot, not sequential resource reservations. Live sampling and compilation
update seed, shed and tile state after each chosen unit action. Public
`market_kind_mask` and `quantity_mask` likewise describe a snapshot, but now share
the live sampler's `MarketLedger` pricing: each product purchase uses its next-unit
quote and quantity affordability uses cumulative quotes, not the displayed price.
Live market sampling retains its post-unit ledger and advances it after each order.

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

The actor trunk's base learning rate defaults to `5e-5` and the critic's to
`1.5e-4` (NorMuon matrices), with `0.35` times each for their ordinary Adam
parameter groups. At the former shared `1.5e-4` every Monte Carlo arm peaked
against its clone by wave 50 and then drifted; `5e-5` held 0.936 argmax against
the clone at the thirty-minute endpoint against 0.68
(`artifacts/probes/ppo-stage2-20260927`). Production
and the direct training CLI default the separate value-head Adam LR to `4.375e-4`, preserving
its `25/3` boost over ordinary Adam groups. The raw training CLI accepts
`--critic-head-lr` as an optional absolute override. Each group retains its own
32-optimizer-step linear LR warmup and checkpointed state. NextLat predictors
inherit their corresponding actor/critic base rate unless explicitly overridden;
the `lejepa` world model and backbone default to their own 1.5e-5.
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

The default physical minibatch ceiling is **4096**, the size every measured
LeJEPA recipe ran and the one the promoted actor rate was calibrated at: a
complete 230080-state production wave uses **57 fixed-shape minibatches**, with no
dropped states or gradient accumulation: 56 full batches and 704 genuine rows in
the last; its remaining 3392 rows have zero loss/gradient weight. With the update
retaining its farm activations (`--no-rematerialize-actor-update`, the `lejepa`
default; every other family keeps replay, its memory without it unmeasured)
the production update peaks at 17.2 GiB and runs 0.36 s per wave faster than
replaying them; 16384 rows do not fit the device
(`artifacts/probes/ppo-speed-20260927`). The former entity-attention ceiling was
7936, chosen for a 64-row padded tail over 29 minibatches. For that family, a
matched six-repeat probe measured
steady whole-iteration medians of 10.952 s at 6400 versus 11.196 s at 8192,
with peak live memory 16.33 versus 19.99 GiB. All intended updates completed.
Larger batches reduce optimizer steps per wave (36 to 29 here) and change gradient
statistics; they are not learning-equivalent merely because sample coverage is
unchanged. Learning rates, objectives, and precision are unchanged. Historical
fixed-shape probes and explicit minibatch overrides retain their declared sizes.

`train_ppo.py --architecture-panel 25` evaluates 64 fixed development seeds
against `starter` and `scripted-v27`, using both sampled and argmax actions on
compiled native BF16 paths. Every evaluation preserves an immutable checkpoint.
After 150 actor-active waves, a run is culled only when its smoothed score is at
least 0.05 below initialization and neither score nor heldout critic fit has
materially improved for 100 actor-active waves. An absolute score increase of
0.01 or critic MSE reduction of 0.01 in the alpha-0.5 EMA resets patience.
Critic MSE uses sampled-policy panel returns and remains defined when every
game has the same outcome; MC R-squared is also reported when target variance
is nonzero. The guard, best evaluated checkpoint and panel policy persist on
resume. It defaults to every 25 actor-active waves for a single learner and off
for a population or with `--autocull`, which it cannot be combined with. It
requires CUDA, a compiled update, terminal-outcome and gamma 1, so it also defaults
off wherever one of those is missing; an explicit interval there is refused.

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
the pinned `kaggle-environments==1.32.7` release. The companion pages cover:

- [agent API and state](mechanics/api.md)
- [default constants](mechanics/constants.md)
- [farm and shed](mechanics/farm.md)
- [farmers and farm hands](mechanics/farmers.md)
- [crops](mechanics/crops.md)
- [animals](mechanics/animals.md)
- [market](mechanics/market.md)
- [town demand](mechanics/town.md)
