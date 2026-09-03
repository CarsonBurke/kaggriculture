# kraggiculture

Research and evaluation tooling for the Kaggriculture simulation competition.

The learner is direct, from-scratch self-play PPO with DAPO's asymmetric clip
band, discount-correct potential shaping, and VAPO's decoupled GAE. Training
does not currently depend on expert demonstrations, distillation, behavior
cloning, or value pretraining. An exact
batched Rust simulator supplies high-throughput rollouts; the pinned Kaggle
environment remains the parity oracle and final evaluator.

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
    --games 128 --repeats 6 "$@" \
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
  --run-dir "$repo/runs/ppo-main"
```

The launcher rejects partial reports or any mismatch in source, hardware, seed,
precision, model, PPO, or data-generation settings.

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
3.62 s. It is also 8.4x further inside the gate it risks: over four production
waves the shipped replay-parity audit measures a worst max_kl of 2.2786e-04
under inductor/bf16 against 1.9089e-03 under eager/fp32, on a bound of 5e-3.
The drift is dominated by systematic differences between the collection and the
update path rather than by rounding, and the update path is already Inductor
plus bf16, so matching it cancels most of the difference. That is also why the
collection precision is held fixed at bf16 on every node instead of becoming a
third knob: three reports attribute two knobs because each step moves exactly
one, and the precision's answer is settled by measurement outside the chain.

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

Training uses discount-correct, exactly zero-sum potential shaping. Let `L[i,t]`
be player `i`'s actual liquid assets: bank money plus the exact proceeds from
selling every held product at the current market curve. With the game-defined
starting bank `k = 3000`,

```
P[t] = log((L[0,t]       + k) / (L[1,t]       + k))  # nonterminal potential
U[T] = log((bank[0,T]    + k) / (bank[1,T]    + k))  # terminal utility

r[0,t] = gamma * P[t+1] - P[t]  # nonterminal
r[0,T-1] = U[T] - P[T-1]        # terminal; terminal shaping potential is zero
r[1,t] = -r[0,t]
```

The production discount is `gamma = 0.997`. From the symmetric initial state
`P[0] = 0`, the discounted complete return is
`gamma^(T-1) * U[T]`. Every game has the same horizon, so this factor cannot
change the ordering of terminal outcomes. The intermediate potential cancels
without requiring gamma one, adding dense credit assignment without an
early-lead or time-average occupancy objective. Every transition, including the
terminal transition, sums to exactly zero.

The starting bank supplies a game-defined zero-asset prior. This keeps the
comparison percentage-like and defined at zero without the extreme slope of a
one-dollar pseudocount: `3000` versus `0` scores `log(2)`, not `log(3001)`.
There is no fitted dollar scale or nonlinear margin saturation.

Liquid assets deliberately exclude seeds, animals, planted crops, pending
yields, and land because the market cannot liquidate them. Market products are
valued by walking the engine's sell arithmetic unit by unit, including its
price-floor restock rule. Moving those products into the bank is therefore
potential-neutral, so cycling inventory cannot manufacture reward.

The log potential is not artificially bounded. Critic targets outside the
categorical support saturate at its outer atom, and the saturated fraction is a
reported training gate rather than a hidden reward transform. Rust supplies
binary32 potentials and terminal utility; one Python reward implementation
applies the same configurable gamma to native and interpreted rollouts.

GAE follows VAPO's decoupled schedule, not CleanRL's shared lambda. Policy
advantages use ``lambda = 1 - 1/(0.05 * 719)`` -- VAPO's length-adaptive
formula evaluated at the known 719-action horizon, not per sequence. Critic
targets use lambda 1, the unbiased discounted suffix return. Actor advantages,
critic targets, and collection shaping all use the same gamma. Targets beyond
categorical support saturate at the outer atom and the saturated fraction is
reported.

The trust region is `target_kl = 0.03`; at the shipped actor learning rate the
population runs measure per-iteration approx KL of 1e-4 to 2e-4, so the region
rarely binds.

Entropy is measured for collapse detection but never optimized: `PpoConfig` and
the training CLI expose no entropy coefficient, and the actor loss is exactly
the clipped PPO surrogate. Population training uses four independently
initialized live learners, uniform ordered round-robin pairings, and both seats'
trajectories. `--population 4` requires `--league-games 0`; validation rejects
frozen snapshots and built-in opponents in the wave. External opponents are
evaluation-only diagnostics and never affect training gradients.

Members are kept far from duplicates in the states they receive. Every game in
a wave has its own seed (`seed_start + g`), so a member's 78 games per wave are
distinct maps, and two members share only their direct head-to-heads. On top of
that each member trains under its own fixed board orientation
(`MEMBER_ORIENTATIONS[member % 4]`): identity, horizontal mirror, vertical
mirror, or a 180-degree rotation. The orientation transforms the encoded board,
the unit positions, and the movement actions consistently -- what the oriented
view calls EAST executes as the real direction the map sends it -- so every
member acts legally in the shared environment while receiving genuinely
different state streams. Checkpoints record each member's orientation and
evaluation and submission replay it; the structured (non-conv) encoding does
not yet carry an orientation mapping and rejects non-identity populations.

The actor and critic have separate spatial U-Nets and fixed-token entity
transformers. The actor attends over one state token, 100 board cells, 16 unit
slots, and 10 autoregressive market slots using pre-normalization, ReLU-squared
feed-forwards, normalized queries/keys, axial RoPE, long U-shaped residual skips,
and PyTorch scaled-dot-product attention (Flash Attention on eligible CUDA
inputs). Inactive unit slots are zeroed after every block; legality is enforced
by the exact sequential action ledger, not leaked into attention. All game
actions are discrete. Market quantities use a state- and order-conditioned
masked categorical over every integer from 1 through 100, which preserves exact
PPO likelihoods and multimodal quantity choices; a continuous Beta density would
not be a valid likelihood for these integer actions. The centralized critic uses
HL-Gauss labels on a bounded categorical support; target saturation is measured
and gated because the log-relative economic return itself is not clipped.

Future ablation, intentionally not implemented yet: actor-only pretraining on
the public v27 route. Demonstrations must first be projected through the exact
sequential legality ledger (invalid unit actions become PASS, invalid market
orders are removed and compacted rather than converted to STOP, and excessive
quantities are clamped to the largest legal integer). Any pretrained actor must
then enter RL with a fresh critic and optimizer, with no persistent BC or KL
term, and must beat the from-scratch initialization on held-out paired seeds.

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
  --seeds 32 --device cuda \
  --output "$repo/evaluations/checkpoint-000100-v27-screen.json"
```

The official 32-seed paired panel is enough to package. Do not re-run 128 seeds
against an artifact you already screened. Rank that panel, not the symmetric
self-play score -- a zero-sum population averages 0.5 internally regardless of
its absolute strength -- or `latest.pt`. Evaluation takes no compilation flag:
`--rollout-forward-mode`, `--rollout-bfloat16` and `--update-compile-mode` are
training knobs, decided by the calibration described above.

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

Package admission is a separate CPU contract matching Kaggle. Run the public
finalist panel on its default seed range (disjoint from selection's 10,000,000
range), then the mandatory built-in `starter` gate:

```bash
PYTHONPATH="$snapshot/src" "$repo/.venv/bin/python" \
  "$snapshot/scripts/evaluate_checkpoint.py" \
  --artifact "$repo/runs/ppo-main/best.pt" \
  --selection-report "$repo/evaluations/ppo-main-screen.json" \
  --seeds 32 --device cpu \
  --output "$repo/evaluations/ppo-main-finalist-v27.json"

PYTHONPATH="$snapshot/src" "$repo/.venv/bin/python" \
  "$snapshot/scripts/evaluate_checkpoint.py" \
  --artifact "$repo/runs/ppo-main/best.pt" \
  --opponent starter --seeds 16 --device cpu \
  --output "$repo/evaluations/ppo-main-starter.json"

PYTHONPATH="$snapshot/src" "$repo/.venv/bin/python" \
  "$snapshot/scripts/build_submission.py" \
  --checkpoint "$repo/runs/ppo-main/best.pt" \
  --evaluation-report "$repo/evaluations/ppo-main-finalist-v27.json" \
  --builtin-evaluation-report "$repo/evaluations/ppo-main-starter.json" \
  --output "$repo/artifacts/kaggriculture-ppo.tar.gz"

PYTHONPATH="$snapshot/src" "$repo/.venv/bin/python" \
  "$snapshot/scripts/validate_submission.py" \
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
