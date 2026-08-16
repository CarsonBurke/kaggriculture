# kraggiculture

Research and evaluation tooling for the Kaggriculture simulation competition.

The learner is direct, from-scratch self-play PPO with DAPO's asymmetric clip
band and VAPO's length-adaptive GAE. Training does not currently depend on
expert demonstrations, distillation, behavior cloning, or value pretraining. An exact
batched Rust simulator supplies high-throughput rollouts; the pinned Kaggle
environment remains the parity oracle and final evaluator.

Install both development and training dependencies before running the complete
test suite:

```bash
uv sync --extra dev --extra train
uv run pytest
```

The first Rust-backed rollout automatically builds the native extension with
`cargo build --release`. Rust traces must pass differential tests against
`kaggle-environments==1.32.6` before they are admitted to training.

Run the complete native correctness gate with:

```bash
cargo test --manifest-path rust/kagg_env/Cargo.toml
cargo clippy --manifest-path rust/kagg_env/Cargo.toml --all-targets -- -D warnings
PYTHONPATH=src .venv/bin/python \
  rust/kagg_env/tests/parity_oracle.py --games 8 --steps 719
PYTHONPATH=src .venv/bin/python rust/kagg_env/tests/binding_safety.py
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
    --cwd "$snapshot" \
    --env PYTHONPATH="$snapshot/src" --env PYTHONDONTWRITEBYTECODE=1 \
    --env CARGO_TARGET_DIR="$repo/artifacts/cargo-target/$digest" -- \
    "$repo/.venv/bin/python" scripts/benchmark_ppo_iteration.py \
    --games 112 --repeats 6 "$@" \
    --output "$repo/artifacts/benchmarks/$name-ppo.jsonl"
}

benchmark eager
benchmark mixed --compile-update
benchmark compiled --compile-rollout --compile-update
```

All three time the production batch from the same source snapshot and differ
only in the compile flags. Launch training from the complete set:

```bash
.venv/bin/python scripts/launch_calibrated_training.py \
  --eager-report artifacts/benchmarks/eager-ppo.jsonl \
  --mixed-report artifacts/benchmarks/mixed-ppo.jsonl \
  --compiled-report artifacts/benchmarks/compiled-ppo.jsonl \
  --run-dir runs/ppo-main
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
whole iteration and is enabled. The rollout knob is worth about 1.006x on the
whole iteration -- 1.027x on its own phase, roughly 0.2 s out of 37 s -- and is
rejected. An earlier two-repeat calibration reported the rollout at about
0.46x, the collector's per-step graph replay supposedly costing more than the
kernel launches it removes; that figure is withdrawn rather than explained. It
does not survive six repeats, where the compiled rollout median is 7.876 s
against an eager 8.200 s in that earlier run and 8.505 s in the current one. A
single blended total would still enable both knobs, since their sum favors
compiling, and the run would carry the rollout's warmup and replay risk to buy
two tenths of a second. Read each run's own decision file for its numbers
rather than these; the point that survives is the shape, not the magnitude.

All three reports time the production batch and nothing else, because the
decision reads the steady medians at 112 games and nothing from the other
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

Training is direct from-scratch self-play. Rewards are the change in a dense
shaping potential: player zero receives it, player one receives the negative.
Mid-episode that potential is the bounded relative farm value
`(value_0 - value_1) / (value_0 + value_1)` (zero when both sides are empty),
where each side is an exact liquidation core -- bank money plus the exact
proceeds of selling every held product, unit by unit at the quotes the engine
would actually pay, so market product trades stay potential-neutral and
harvested-but-unsold output is credited at true sale proceeds -- plus a
heuristic cost-basis credit for the assets the market cannot buy back: seeds at
0.85 of engine cost in the shed and 0.8 once planted, animals at 0.82 in the
shed and 0.85 once placed, pending yields at 0.72 of their posted price, and
each extra unlocked tile at 0.9 of its land price. Those credits exist only to
smooth credit assignment across the invest-produce-sell loop; without them
self-play collapses into a never-spend tie equilibrium before harvests can pay
back, and kept near engine cost they make a purchase a small potential dip
rather than a shaped-reward cliff the policy never crosses. The terminal
transition then deliberately overrides the potential with the relative scored
bank `(money_0 - money_1) / (money_0 + money_1)`, the quantity the engine
actually scores, so the heuristic weights cannot move the objective.

Because the reward is a potential difference and gamma is fixed at 1.0, the
shaped suffix return from any state telescopes exactly to the final relative
bank score minus that state's potential. The start is symmetric, so its
potential is zero and the undiscounted episode return is exactly the final
normalized bank margin, whose sign exactly matches the game's winner/tie
relation. Gamma is not free to tune here: discounting potential differences
would introduce a separate preference for holding cash early. The telescoping
also means that return's variance is mostly the game's own coin flip rather
than anything the critic can read at a single state, so explained variance
against it sits at or near zero even for a healthy critic. That is why it is
reported as `monte_carlo_explained_variance` rather than as the critic's
accuracy, and why the critic's own regression is scored beside it twice, as
`critic_fit_explained_variance_first_epoch` and
`critic_fit_explained_variance_last_epoch`. Both read the clipped target the
critic actually regresses on, against predictions taken from inside the
update, where `monte_carlo_explained_variance` and
`lambda_return_explained_variance` both read the unclipped return. There are
two because a single in-sample reading cannot separate a critic that is
fitting from one that is memorizing the batch. The last-epoch reading is the
in-sample one deliberately: it scores every state on its final pass, already
fitted three times over at the four critic epochs production configures, while
the first scores those same states before this update has touched them. A
critic that generalizes keeps the two together and one that memorizes the
batch pulls them apart. The epochs between are never mixed in, and with a
single configured critic epoch the two coincide, that epoch being both.

Advantages use VAPO's length-adaptive
`lambda_policy = 1 - 1 / (0.05 * 719) = 0.9721835883`, and critic targets are
the matching lambda-return `advantage + value`. VAPO decouples the critic onto
lambda one because its setting pays a single terminal reward, so nothing short
of the whole trajectory is unbiased; here the reward is a dense potential
difference at every transition, and the lambda-one suffix return would instead
fold the noise of all 719 actions into every earlier state's target. Targets
are saturated at the outermost value atom, as any categorical critic must be
once it bootstraps, and the saturated fraction is reported. Valid games
always contain 719 actions, so the paper's length-adaptive formula is constant
for this environment. Epoch permutations are split into balanced minibatches
so a short tail cannot receive a disproportionate optimizer step. No entropy
bonus is involved. The production defaults collect 112 live self-play games
plus 96 frozen-league games and replay them once. Frozen
actor-only snapshots are written every update, with opponents drawn from a
16-policy recent window and log-age historical strata. Recent opponents are
sampled at temperature 0.8; the initial anchor and historical policies use the
same deterministic decoding as a submission. Full resumable checkpoints are
kept every five updates.

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
HL-Gauss labels on a bounded categorical support with headroom around the proven
`[-2, 2]` return range.

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
beside the run artifacts. A portable resume copies and validates the sidecar
before taking another step, and rejects changed source, rollout, optimizer, or
league settings instead of silently forking the data distribution:

```bash
.venv/bin/python scripts/train_ppo.py \
  --run-dir runs/ppo-resumed \
  --resume runs/ppo-main/checkpoint-000100.pt
```

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
orientations. The default opponent is the fixed public v27 reference; any
failed, truncated, or non-finite game invalidates the result instead of being
silently excluded:

```bash
.venv/bin/python scripts/evaluate_checkpoint.py \
  --artifact runs/ppo-main/checkpoint-000100.pt \
  --seeds 32 --workers 12 \
  --output evaluations/checkpoint-000100-v27.json
```

Use at least `--seeds 128` for finalists. Rank stable-panel results, not the
training self-play score (which is 0.5 by symmetry) or `latest.pt`. Evaluation
takes no compilation flag: `--compile-rollout` and `--compile-update` are
training knobs, decided by the calibration described above, and the evaluation
and selection scripts accept neither.

To screen every numbered checkpoint on identical paired seeds and atomically
promote the strongest lower-confidence-bound result:

```bash
.venv/bin/python scripts/select_checkpoint.py \
  --run-dir runs/ppo-main --seeds 32 --workers 12 \
  --output evaluations/ppo-main-screen.json \
  --best-output runs/ppo-main/best.pt

.venv/bin/python scripts/evaluate_checkpoint.py \
  --artifact runs/ppo-main/best.pt \
  --selection-report evaluations/ppo-main-screen.json \
  --seeds 128 --workers 12 \
  --output evaluations/ppo-main-finalist-v27.json
```

Build a submission only from a selected, evaluated checkpoint:

```bash
.venv/bin/python scripts/build_submission.py \
  --checkpoint runs/ppo-main/best.pt \
  --evaluation-report evaluations/ppo-main-finalist-v27.json \
  --output artifacts/kaggriculture-ppo.tar.gz

.venv/bin/python scripts/validate_submission.py \
  --archive artifacts/kaggriculture-ppo.tar.gz \
  --opponent v27 --seeds 2 \
  --output evaluations/kaggriculture-ppo-bundle.json
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
