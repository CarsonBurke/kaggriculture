# kraggiculture

Research and evaluation tooling for the Kaggriculture simulation competition.

The learner is direct, from-scratch self-play with VAPO-style factor clipping
and decoupled GAE. Training does not currently depend on expert
demonstrations, distillation, behavior cloning, or value pretraining. An exact
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

```bash
snapshot=$(.venv/bin/python scripts/freeze_source.py | \
  .venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["source_root"])')
digest=${snapshot##*/}

mlq submit --name kagg-vapo-benchmark --max-parallel-runs 1 \
  --cwd "$snapshot" \
  --env PYTHONPATH="$snapshot/src" --env PYTHONDONTWRITEBYTECODE=1 \
  --env CARGO_TARGET_DIR="$PWD/artifacts/cargo-target/$digest" -- \
  "$PWD/.venv/bin/python" scripts/benchmark_vapo_iteration.py \
  --games 64,112,128 --output "$PWD/artifacts/benchmarks/eager-vapo.jsonl"
```

Run the matched compiled benchmark from the same snapshot, then launch
`scripts/launch_calibrated_training.py` with both reports. The launcher rejects
partial reports or any mismatch in source, hardware, seed, precision, model,
VAPO, or data-generation settings. It enables compilation only for a measured
steady-state speedup of at least 1.05x and binds the complete decision into
every checkpoint.

Training is direct from-scratch self-play. At every transition, player zero's
potential is the exact relative scored-bank value
`(money_0 - money_1) / (money_0 + money_1)` (zero when both banks are empty),
and rewards are its change; player one receives the negative. There is no
terminal reward override and no guessed mark-to-market value for inventory,
land, animals, or future production. Since both players start with equal money,
the undiscounted episode return is exactly the final normalized bank margin,
whose sign exactly matches the game's winner/tie relation. Gamma is fixed at
1.0: discounting potential differences would introduce a separate preference
for holding cash early. Following VAPO's decoupled GAE, actor advantages use
`lambda_policy = 1 - 1 / (0.05 * 719) = 0.9721835883`, while critic targets use
lambda one and are therefore exact Monte Carlo suffix returns. Valid games
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
.venv/bin/python scripts/train_vapo.py \
  --run-dir runs/vapo-resumed \
  --resume runs/vapo-main/checkpoint-000100.pt
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
  --artifact runs/vapo-main/checkpoint-000100.pt \
  --seeds 32 --workers 12 \
  --output evaluations/checkpoint-000100-v27.json
```

Use at least `--seeds 128` for finalists. Rank stable-panel results, not the
training self-play score (which is 0.5 by symmetry) or `latest.pt`. Model
compilation is deliberately opt-in with `--compile-models` until the queued
eager/compiled numerical and throughput comparison proves it beneficial on the
target GPU.

To screen every numbered checkpoint on identical paired seeds and atomically
promote the strongest lower-confidence-bound result:

```bash
.venv/bin/python scripts/select_checkpoint.py \
  --run-dir runs/vapo-main --seeds 32 --workers 12 \
  --output evaluations/vapo-main-screen.json \
  --best-output runs/vapo-main/best.pt

.venv/bin/python scripts/evaluate_checkpoint.py \
  --artifact runs/vapo-main/best.pt \
  --selection-report evaluations/vapo-main-screen.json \
  --seeds 128 --workers 12 \
  --output evaluations/vapo-main-finalist-v27.json
```

Build a submission only from a selected, evaluated checkpoint:

```bash
.venv/bin/python scripts/build_submission.py \
  --checkpoint runs/vapo-main/best.pt \
  --evaluation-report evaluations/vapo-main-finalist-v27.json \
  --output artifacts/kaggriculture-vapo.tar.gz

.venv/bin/python scripts/validate_submission.py \
  --archive artifacts/kaggriculture-vapo.tar.gz \
  --opponent v27 --seeds 2 \
  --output evaluations/kaggriculture-vapo-bundle.json
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
