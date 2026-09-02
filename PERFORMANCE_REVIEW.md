# Performance Review

## Scope and method

This review covers the production PPO loop, native Rust simulator and Python binding,
population training, evaluation/submission inference, behavior-cloning data loading, and
checkpoint/telemetry I/O. Generated runs and benchmark artifacts were used as evidence; they
were not reviewed as implementation code.

The review used four kinds of evidence:

1. End-to-end timing records in `artifacts/benchmarks/` and `evaluations/`.
2. Stage profiles and parameter sweeps in `artifacts/benchmarks/`, `artifacts/sweeps/`, and
   `artifacts/probes/`.
3. An unarchived local scalar-core spot measurement from:
   `cargo run --manifest-path rust/kagg_env/Cargo.toml --release --bin bench_env -- 4096`.
   It is supporting context, not durable benchmark evidence.
4. Static tracing of every hot path and its allocation, transfer, synchronization, and
   scaling behavior.

The archived PPO benchmark is a shipped-snapshot baseline, not a fresh measurement of the
current tree. Its embedded source hashes differ from the current `ppo.py`, `rollout.py`,
`production.py`, and `train_ppo.py`. It is still the best matched phase attribution in the
repository, but every proposed change must be re-baselined on one frozen current source tree.

Priority meanings:

- **P0:** measured material bottleneck with a concrete, low-ambiguity next experiment.
- **P1:** material bottleneck or scaling limit; implementation should follow a focused
  benchmark or quality gate.
- **P2:** conditional or lower-impact opportunity. Do not schedule ahead of P0/P1 work.

## Executive summary

The shipped matched benchmark records a 25.207 s steady production iteration on the RTX 5090:

| Phase | Median | Share of total |
|---|---:|---:|
| PPO update | 18.818 s | 74.7% |
| Mixed rollout | 6.279 s | 24.9% |
| Opponent setup | 0.049 s | 0.2% |
| Total | 25.207 s | 100% |

Source: `artifacts/benchmarks/ship-compiled-bf1ec01b8fb3080c6beace05d4ad78fc6df15468d04cdaea36bd55957f61af2a.jsonl`.
The small phase/total discrepancy is expected because the report uses medians independently.

Compilation has already captured the obvious win. In the same matched chain:

| Configuration | Rollout | Update | Total | Iterations/hour |
|---|---:|---:|---:|---:|
| eager rollout, eager update | 13.060 s | 45.932 s | 59.030 s | 60.99 |
| eager rollout, compiled update | 13.515 s | 18.649 s | 32.106 s | 112.13 |
| compiled rollout, compiled update | 6.279 s | 18.818 s | 25.207 s | 142.82 |

The next work should therefore remove work, not toggle another generic compiler flag.

| Rank | Bottleneck | Evidence | Recommended direction |
|---|---|---|---|
| P0 | Second steady-state critic epoch | A historical probe extrapolates 4.6 s per production-sized epoch and finds worse warm-critic holdout loss on epoch 2 | Use a phase-aware 3/1 or 2/1 fresh/steady critic schedule after a current-tree longitudinal A/B |
| P1 | Heavy critic architecture | Full CNN plus transformer is replayed and trained while game-held-out EV remains 0.5–2% | Benchmark a smaller centralized critic; rank by time-to-score |
| P1 | Batch-one evaluation action path | A historical 64-game panel took 116.0 s; complete candidate actions account for at least 64.6% of wall time under ideal four-worker scaling | Stage-profile encoding/forward/decoding, then test lockstep batching or native screening |
| P1 | BC corpus re-encoding | Approximately 10 MiB/episode-seat and 20 GiB for 2,048 seats, rebuilt every launch | Add provenance-keyed encoded shards and stream into final arrays |
| P1 | Population rollout restaged once per member | Full wave upload/scans are inside each sequential member update | Stage immutable rollout tensors once per iteration |
| P1 | Wasted built-in/frozen league work | Built-in rows are encoded and can receive fake neural forwards whose outputs are discarded | Make row roles explicit and execute only neural rows in the ensemble |
| P2 memory | Dense action masks | 506.4 MiB per production wave, about 15% of persisted conv rollout storage | Bit-pack only if measured memory headroom or end-to-end time justifies device unpacking |
| P2 | Native encode/sample/output work | Archived current-generation profile attributes 52% of a pure rollout step to Rust encode plus sample/step | Remove duplicated score work, share per-game encoding, then test subwave overlap |

## Findings

### 1. P0: a historical probe identifies the second critic epoch as likely overspend

**Evidence**

Production explicitly configures one actor epoch and two critic epochs in
`src/kaggriculture/production.py:99-146`. The underlying measurements are
`artifacts/probes/probe-critic-epochs-fresh-6fa2cd2f559e.json`,
`artifacts/probes/probe-critic-epochs-warm-6fa2cd2f559e.json`, and
`artifacts/probes/probe-critic-epochs-warm6-6fa2cd2f559e.json`. They bind source digest
`6fa2cd2f559efcd4c464d2708737c8279326130ec08428b8fa57df9b46d8b6dd`, use 112
self-play games rather than the mixed production wave, and rescale per-epoch time from a
smaller fit set to 113 production minibatches. Their quality findings are nevertheless strong:

- Game-level holdout explained variance is only 0.5–2% for every tested epoch count from 1 to 8.
- After the critic is warm, epoch 1 changes holdout distributional loss by
  -0.081 +/- 0.049; epoch 2 changes it by +0.072 +/- 0.021, giving back slightly more than
  epoch 1 gained.
- A fresh critic prefers three epochs; a warm critic prefers one.
- The probe extrapolates approximately 4.6 s per production-sized epoch and 9.2 s for two.

The loop that pays this cost is `src/kaggriculture/ppo.py:2188-2401`. Applying the historical
4.6 s estimate to the archived 25.207 s production baseline yields an estimated 18%
whole-iteration and 24% update-phase opportunity. These percentages are planning estimates, not
a current-tree measurement.

**Problem**

One static compromise serves two different regimes. Early training needs extra critic fitting;
steady training repeatedly fits the same rollout after the game-level holdout evidence says the
second pass is harmful. The current comment retains it only as insurance against long-horizon
tracking failure, not because the second same-wave pass has demonstrated value.

**Proposal**

Add an explicit per-call critic-epoch override and use a phase-aware schedule:

- Fresh critic / critic-only warmup: measure two versus three epochs.
- Actor-active steady state: one critic epoch.
- Optional temporary second epoch only when a next-wave quality metric shows the critic falling
  behind. Do not trigger from same-wave fit, which rewards memorization.

The boring first experiment is fixed `3 during warmup, 1 after warmup` versus the current fixed
`2`, not an adaptive controller. Add adaptation only if the fixed schedule exposes a real tracking
failure.

**Acceptance benchmark**

Run matched, multi-seed 100–500 iteration A/Bs with identical rollout seeds and actor settings.
Compare:

- true wall time and time-to-external-score;
- next-wave critic loss/EV, never only same-wave fit EV;
- policy entropy, final economy, and score rate against held-out opponents;
- run-stopping gates and value-target saturation.

Do not merge on the 4.6 s saving alone. The actor consumes these advantages, so learning quality
is the contract.

### 2. P1: the critic spends heavily without demonstrated generalization

**Evidence**

`DistributionalCritic` owns a separate `SpatialUNet`, projects 101 centralized features, and
runs an `EntityTransformer` over one state token plus all 100 board tokens before reading only
the state token (`src/kaggriculture/model.py:745-788`). Every update pays for:

1. a full behavior-value replay before GAE (`src/kaggriculture/ppo.py:2072-2085`);
2. one or two critic forward/backward passes over every valid state
   (`src/kaggriculture/ppo.py:2205-2395`).

The historical probe cited above finds only 0.5–2% game-held-out EV even while same-wave fit EV
climbs to 0.87. The update is 74.7% of the archived iteration, and the probe extrapolates 4.6 s
for one production-sized critic pass. This makes critic capacity the largest architectural
performance candidate after the epoch count is re-measured on the current tree.

**Proposal**

Benchmark a centralized critic family independently of the actor:

- pooled spatial encoder plus an economy/state MLP;
- fewer/narrower transformer layers;
- state-token cross-attention into spatial features instead of repeated full self-attention over
  101 tokens.

Keep centralized private inputs and the distributional support. Do not share the actor trunk by
default: the actor is decentralized, and shared policy/value gradients introduce a learning
coupling that this performance change does not require.

**Acceptance benchmark**

First compare 2,048-row device time, peak allocation, and replay throughput. Then run the existing
game-level holdout probe and matched PPO learning curves. Rank candidates by time-to-score and
next-wave critic quality, not parameter count or same-wave loss.

### 3. P1: evaluation runs the complete candidate action path at batch size one

**Evidence**

`CheckpointAgent.__call__` always calls `act_batch` with a one-element list
(`src/kaggriculture/inference.py:232-268`). `evaluate_checkpoint.py` has two execution modes:

- accelerator evaluation is restricted to one worker;
- CPU parallel evaluation spawns independent workers, each with a complete model, but every
  action remains batch size one (`scripts/evaluate_checkpoint.py:224-244,280-300,577-590`).

The historical report `evaluations/econ-pastself-lr1e5-i57-v27.json` records:

- 64 games, four CPU workers, 116.007 s elapsed;
- 46,016 complete candidate action calls at 6.517 ms mean;
- 299.9 aggregate CPU-seconds in the candidate action path.

Its embedded source hashes differ from the current evaluator, inference, and policy files. The
timer wraps the complete `_WORKER_AGENT(observation)` call: observation encoding, model forward,
device transfers, sequential mask/action decoding, and weed cleanup. It does not isolate neural
inference. Even with perfect four-worker scaling, this complete path accounts for at least
75.0 s, or 64.6% of wall time. A stage profile is required to determine how much lockstep
batching can remove.

Current CUDA code also performs avoidable per-action boundary work: `act_batch` copies three actor
outputs and the three immutable quantity-head tensors to CPU on every call before sequential
NumPy/Python masking (`src/kaggriculture/policy.py:510-515`). The quantity-head weights cannot
change inside a `CheckpointAgent`, but their share of action latency is not yet measured.

**Proposal**

Implement in increasing order of ambition:

1. Cache immutable quantity-head NumPy arrays in an inference-only prepared sampler. Preserve the
   mutable actor path used by training.
2. Advance multiple official environments in lockstep and call `act_batch` once for the full
   observation batch. One process should own the accelerator model; do not create one CUDA
   context/model per environment worker.
3. For checkpoint screening, add a native `BatchEnv` evaluator for opponents already ported to
   Rust. Keep the pinned official environment as the final parity/evidence gate.

An unarchived local spot measurement completed 4,096 PASS games in 0.247011 s
(16,582 games/s). It excludes policy inference, masks, encoding, built-ins, and binding traffic,
and it is not bound to a persisted source-identity record. It suggests raw native stepping has
headroom; it is not durable evidence or an end-to-end throughput prediction.

**Acceptance benchmark**

Use a fixed 128-seed/two-seat panel. Compare current CPU 1/2/4/8 workers, current CUDA, lockstep
CUDA batches 8/32/128, and native screening. Require identical actions and game results where the
same engine is used, official/native parity on the final subset, bounded per-action p99, RSS per
process, and end-to-end elapsed time.

### 4. P1: behavior-cloning launch repeatedly rebuilds a 20 GiB encoded corpus

**Evidence**

Every `train_bc.py` launch opens each compressed NPZ, decompresses `raw_json_zlib`, parses JSON,
and reruns the architecture encoder for every observation
(`scripts/train_bc.py:334-362`). Worker results are fully materialized in a list before the train
and holdout splits are staged (`scripts/train_bc.py:513-536`).

The source records the measured scale at `scripts/train_bc.py:378-386`:

- approximately 10 MiB per episode-seat;
- approximately 20 GiB steady state for 2,048 seats;
- an earlier concatenate implementation peaked near 40 GiB and was killed.

Preallocation fixed the 2x peak, but identical experiments still pay decompression, JSON parsing,
feature encoding, process-pool transfer, and the 20 GiB resident corpus every launch.

**Proposal**

Create architecture- and provenance-bound encoded shards:

- cache key: dataset manifest digest, source/tokenizer schema identity, architecture and feature
  dtypes;
- contiguous or memory-mapped shards rather than thousands of process-pickled episode dicts;
- atomically publish complete shards;
- stream shards directly into the final split layout rather than first collecting every worker
  result in `encoded`.

Keep raw observations as canonical evidence. The cache is derived data and must be rejected when
its schema/source identity does not match.

**Acceptance benchmark**

On a representative 2,048-seat corpus, measure cold and warm startup to epoch 0, max RSS, disk
read volume, CPU time, and cache size. Require shape/dtype equality, byte digests for every staged
field, and matching epoch-0 metrics against the uncached loader.

### 5. P1 in population mode: immutable rollout staging scales with member count

**Evidence**

Population members update sequentially in `scripts/train_ppo.py:2320-2354`. Each call enters
`update_ppo`, where every state, action, and dense mask array for the complete rollout is uploaded
again (`src/kaggriculture/ppo.py:2053-2071`). The `rows` restriction is applied only after that
full staging operation.

For population size N, each member owns roughly 1/N of the states, but full-wave host scans and
H2D staging are repeated N times. Model compute remains member-specific; immutable rollout
transport should not.

**Proposal**

Introduce an iteration-scoped `StagedRollout`:

- upload immutable state/action/mask tensors once;
- retain per-member row indices;
- create only member-specific behavior replays, advantages, and targets per update;
- release the shared staged object after all members finish.

If one shared GPU staging object raises peak memory, the fallback is a one-time host partition
into member-major contiguous buffers. Total copied bytes should be one rollout, not N rollouts.
Per-member advantage normalization and replay parity remain unchanged.

**Acceptance benchmark**

Add a population cost probe for N=1/2/4/8. Record H2D bytes, staging seconds, cold compile time,
steady update time, and peak reserved memory. Require matching per-member metrics, actions, and
checkpoint-resume behavior.

### 6. P1: league collection executes rows that cannot affect actions or storage

**Evidence**

Production reserves three lanes among four native built-in opponents
(`src/kaggriculture/production.py:19-36`) and collects 96 league games per wave. In
`collect_mixed_play_rust`:

- built-in lanes borrow frozen network weights so the ensemble shape remains static;
- their neural outputs are explicitly discarded;
- all active lanes are padded to the widest group
  (`src/kaggriculture/rollout.py:1192-1231,1257-1297`).

The binding also encodes both seats for every game
(`rust/kagg_env/src/python.rs:1180-1241`) and computes full factor masks/statistics for built-in
rows (`rust/kagg_env/src/python.rs:484-572`), although only learner rows are stored
(`src/kaggriculture/rollout.py:1187-1191,1311-1337`). Sample outputs are copied densely at
`rust/kagg_env/src/python.rs:1354-1392`. This is deliberate graph-shape stability, but it trades
recurring work for avoided shape variants without a current break-even measurement.

**Proposal**

Pass explicit row roles to the binding and collector:

- learner row: encode, run network, sample, and store;
- frozen neural row: encode, run the actual frozen ensemble, sample, do not store;
- built-in row: compute built-in action once, skip neural encoding/forward and unneeded sampled
  outputs.

Build the frozen ensemble only from actual neural lanes. Cache compiled callables by neural lane
count and padded width. The existing cache already keys compiled ensemble calls by mode and width
(`src/kaggriculture/rollout.py:811-843`), so fake built-in lanes should not be the only way to
manage shape variation.

**Acceptance benchmark**

Sweep 0/25/50/100% built-in league seats and balanced/skewed assignments. Attribute encode,
transfer, learner forward, frozen forward, native sample/step, and total rollout time. Require
identical learner fields, built-in actions, stateful v27 progression, RNG consumption, and final
money.

### 7. P2 memory: dense masks dominate action metadata, not total rollout storage

**Evidence**

Every trajectory-step stores:

- 16 x 68 unit-mask booleans;
- 10 x 22 market-kind booleans;
- 10 x 100 market-quantity booleans.

The fields are byte-sized `np.bool_` arrays (`src/kaggriculture/rollout.py:183-198,273-315`) backed
by dense Rust `bool` arrays (`rust/kagg_env/src/core.rs:382-414`). That is 2,308 bytes per
trajectory-step.

The production arena has 320 learner trajectories (224 self-play plus 96 league) and a 719-step
horizon:

`320 * 719 * 2,308 = 531,024,640 bytes = 506.4 MiB`

For comparison, the conv-entity board alone is 58 x 10 x 10 fp16 values, or 11,600 bytes per
trajectory-step and 2,545.3 MiB for the same arena. Dense masks are therefore about 15% of
persisted conv rollout storage, not the dominant field. The 443 MiB potential saving is
conditional on memory headroom or measured end-to-end benefit.

This excludes active flags, scratch buffers, state features, and duplicated staging. The binding
copies all dense masks into NumPy output, and Python copies selected rows into the full-horizon
arena (`rust/kagg_env/src/python.rs:1354-1382`, `src/kaggriculture/rollout.py:957-990`).

**Proposal**

Keep current-step dense masks inside the sampler, but bit-pack masks when persisting them. The
three packed rows require 289 bytes per trajectory-step, reducing production mask storage to
63.4 MiB and saving approximately 443.0 MiB. Unpack only selected update minibatches on device,
or unpack once during whole-rollout staging if that is faster.

Do not use `np.packbits` as a post-processing pass over another full dense arena. Pack directly at
the Rust/storage boundary.

**Acceptance benchmark**

Measure end-to-end rollout plus update, not storage size alone. Record arena RSS, bytes H2D,
`_store_native_wave` time, device unpack time, and peak CUDA memory for 32/112/224 self-play
games. Require byte-identical unpacked masks and identical replay logprobs/KL.

### 8. P0: the rollout is launch-bound, and the GPU is idle 84% of it

**This supersedes the earlier reading of this finding**, which used a conv-architecture stage
profile from a different snapshot to conclude that native encode plus sample/step was 52% of a
collection step and therefore "the rollout target". Measured directly on the production
structured wave, that is wrong by a wide margin, and it pointed the work at the wrong half of the
step. (The functions that entry named, `economic_scores` and `training_rewards`, no longer exist
in the Rust tree either.)

**Evidence**

`probe_rollout_utilization.py` runs the full 720-step, 192-game production wave and compares the
device time the profiler attributes to CUDA kernels against the wall time of the same wave:

| | |
|---|---:|
| Wall | 21.685 s |
| GPU busy (union of kernel intervals) | 3.395 s |
| **GPU busy fraction** | **15.7%** |
| GPU idle | 18.290 s |
| Kernel launches | 1,201,642 |
| Launches per step | 1,669 |
| Mean kernel duration | 2.83 us |
| Gaps between kernels | 1,102,592 |
| Gaps under 100 us | 1,093,612, totalling 14.377 s |
| Median gap | 10.24 us |

The gap *distribution* is what identifies the cause. Host-side Rust work would show up as a few
large gaps per step -- roughly 720 of them. Instead there are 1.1 million gaps, 99.2% of them
under 100 us, with a 10.24 us median that is exactly launch-overhead sized. Decomposing the
30.1 ms step: 4.7 ms of GPU work, about 20 ms of sub-100 us launch gaps, and about 5.4 ms in the
larger gaps that contain the native work. Native work is therefore roughly 18% of a step, not
52%, and launch overhead is about two thirds of it.

A mean kernel duration of 2.83 us is the same statement from the other side: at 1,669 launches per
step for a 967k-parameter actor, the device is being fed work in pieces far too small to cover the
cost of asking for them. The largest single device item is `Memcpy DtoH (Device -> Pinned)` at
10,785 calls and 543 ms, about 15 device-to-host copies per step.

Caveat: profiling inflates the step (30.1 ms here against roughly 27 ms unprofiled), so treat the
absolute milliseconds as directional and the ratios as the finding.

**Proposal**

This is the workload CUDA graphs exist for. Capture collapses a whole step's launches into one
replay, which is the only lever that addresses two thirds of the step rather than a slice of it;
every native-side saving is a slice of the remaining 5.4 ms and should wait behind it.

*Why the off-the-shelf route does not work here.* `torch.compile(mode="reduce-overhead")` reaches
capture through `torch._inductor.cudagraph_trees`, which is structurally at odds with a two-shard
collector. It holds a tree manager per **thread** (`cudagraph_trees.py:328`) but decides generation
boundaries from a **process-global** counter, `MarkStepBox.mark_step_counter`, which
`cudagraph_mark_step_begin` decrements and every manager reads through `get_curr_generation`
(line 2547) and `can_start_new_generation` (line 2556). With both shards marking once per step,
the peer's mark lands between a shard's actor forward and its ensemble forward, the manager
concludes the generation ended, and it recycles the actor output the step has not consumed yet:
`accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`, raised from
the `index_copy_` that reads it. `rollout._cuda_graph_generation` repairs exactly that by holding
a lock across a shard's mark and all of its forwards. The mode then still wedges -- every thread
in `futex_wait` at a few percent CPU -- and two attempts cost twenty-five minutes of idle
exclusive GPU between them and produced no stack. It is retired rather than debugged further.

*What replaces it.* `rollout._CapturedStep` captures one `torch.cuda.CUDAGraph` per shard over the
step's forward region and replays it for the remaining 719 steps. Nothing then decides on our
behalf when a recording is retired. This is available because the collector already holds every
input at a fixed address for the life of a wave: `copy_to_device` refreshes persistent device
buffers in place, `_select_inputs` and `_lane_view_inputs` gather into persistent `out=` storage,
every row-index tensor is built once before the step loop, and `_pipeline_replica` refills the peer
shard's weights through `load_state_dict` rather than rebuilding the module.

Feasibility was settled in isolation first, deliberately, because the two full-wave attempts above
were an expensive way to ask a mechanical question. `probe_thread_capture.py` runs two threads on
two side streams with one graph each over the real actor at production shapes, under the same
`inference_mode` and bf16 autocast the collector uses:

| | |
|---|---:|
| Eager forward, two shards contending | 10.537 ms |
| Graph replay | 5.998 ms |
| Speedup | 1.757x |
| Max absolute drift vs eager | **0.0** |

Capture owes bitwise equality rather than the semantic bound the compiled modes settle for, since
it replays the identical kernels over the identical buffers.
`test_a_captured_collection_reproduces_the_eager_one_exactly` holds the mode to that on a wave
carrying both self-play and league rows -- exact equality on actions, logprobs and rewards.

*Measured.* Paired probes chained with `--after-terminal` so both arms saw the same contention:

| Mode | Rollout wall | Per step |
|---|---:|---:|
| eager | 19.810 s | 27.514 ms |
| `graph` | **8.498 s** | **11.803 ms** |

End to end that is 47.507 s -> 33.368 s and 75.78 -> 107.89 iterations an hour; see the resolved
section above. `PRODUCTION_ROLLOUT_FORWARD_MODE` is now `"graph"`.

*Memory.* Capture adds about 64 MiB an iteration of peak allocation over eager for the first eight
collections, and is then exactly flat -- five consecutive repeats at +0.0 MiB, plateauing at
13.937 GiB. That is the league pool filling to its eight opponents plus allocator steady state, not
the private pool accumulating. The distinction mattered: four repeats cannot tell a plateau from a
leak, and a leak of that size exhausts a 32.6 GiB card near iteration 160 of a 500-iteration run,
so the arm was deliberately carried to 14. `_CapturedStep.close()` drops the captured outputs before
resetting the graph, since they live inside the private pool; it makes the lifetime explicit but it
did not change this trajectory, and should not be credited with it.

## Conditional opportunities

### Update backend: small steady win, large cold-start bill

`artifacts/benchmarks/update-backends-e78f3c6b633434ec1fa7dbe625568b3764d4fae67b07eb2898942da8ba7da142.json`
records:

| Mode | Compile | Steady update | Peak reserved |
|---|---:|---:|---:|
| eager | 0.8 s | 41.444 s | 11.34 GiB |
| default | 34.3 s | 17.107 s | 11.34 GiB |
| reduce-overhead | 29.2 s | 17.026 s | 8.44 GiB |
| max-autotune-no-cudagraphs | 204.4 s | 16.377 s | 16.76 GiB |
| max-autotune | 522.6 s | 16.152 s | 8.43 GiB |

This is not the current production schedule, but it establishes the trade. Against `default`,
max-autotune-no-cudagraphs breaks even after about 233 uninterrupted iterations; max-autotune
breaks even after about 512. Resumes and short probes lose. Recalibrate on the final critic
schedule and include compile time in total job wall time before changing defaults.

## Resolved in the attention/optimizer pass

Measured with one frozen tree per arm at production settings
(`scripts/benchmark_ppo_iteration.py`, 128 self-play + 64 league games, 2 epochs, 4 critic epochs,
minibatch 4,096, RTX 5090, medians over the steady repeats). The final arm is a same-tree eager
control, so its rollout column carries the attention regression rather than hiding it.

| Arm | Rollout | Update | Total | Iterations/hour |
|---|---:|---:|---:|---:|
| baseline (`a39b32c6`) | 18.540 s | 41.617 s | 60.272 s | 59.73 |
| attention only (`e83afdc8`) | 19.479 s | 36.181 s | 55.911 s | 64.39 |
| attention + optimizer (`b26874f7`) | 20.383 s | **26.977 s** | 47.507 s | 75.78 |
| + captured rollout (`892038fa`) | **7.270 s** | 25.979 s | **33.368 s** | **107.89** |

Against the baseline: rollout -60.8%, update -37.6%, total -44.6%, throughput **1.81x**. Parity is
neutral throughout: `update_replay_max_kl` moves 3.12e-7 -> 7.81e-7 against a 5e-3 gate, every tail
fraction stays zero, and the first-minibatch KL peaks at 5.97e-7 against a 1.1e-1 gate.

The rollout column is worth reading in order. It *rises* through the first two arms, because the
explicit attention path trades unfused kernels for a fusion the eager collector has no compiler to
perform, and then falls by a factor of 2.8 once capture removes the launches those kernels cost.
The update column also rules out contention as an explanation for the last arm: it is device-bound
and compiled, the rollout mode cannot reach it, and at 25.979 s it lands slightly *below* its own
eager control rather than above.

### Attention ran flash at shapes flash is wrong for

`scaled_dot_product_attention` was the update's largest single kernel: a compiled production
minibatch spent 32.9 ms of its 87.4 ms actor forward+backward inside `_flash_attention_backward`
against a 3.0 ms forward, an 11x backward-to-forward ratio. `model_dim` 80 over four heads is
20 wide, so no fused kernel accepts a masked call and it silently decomposes to the math backend.
`EXPLICIT_ATTENTION_SCORE_LIMIT` in `src/kaggriculture/structured.py` now routes by score count:
an explicit matmul/softmax/matmul that the surrounding Inductor graph fuses below roughly 32M
score elements, and a head-padded cuDNN call above it. Flash was not the fastest option at any
measured geometry. The explicit path costs the *eager* rollout about 0.9 s, because six unfused
kernels beat one flash call only when a compiler is there to fuse them -- which is why the rollout
column rises while the total falls.

### The NorMuon step was 12.24 s an iteration and had never been timed

`profile_update_backends.py` and `sweep_update_batch.py` both time fused AdamW, while production
runs `--optimizer normuon`. Its step was a Python loop issuing roughly 28 kernels per matrix
inside `polar_express` alone, over 110 actor and 100 critic matrices, 342 times an iteration.
Those matrices occupy only 19 distinct shapes with 85 of the actor's 110 in four groups, so
stacking each shape group into one batched Polar Express and moving the Adam half onto
`torch._foreach_*` collapses it:

| Step | Per-parameter | Batched | Speedup | Per iteration |
|---|---:|---:|---:|---:|
| actor (114x) | 34.087 ms | 6.582 ms | 5.18x | 3.886 -> 0.750 s |
| critic (228x) | 36.650 ms | 8.850 ms | 4.14x | 8.356 -> 2.018 s |

Equivalence was established four ways: bitwise-identical CPU results across six scenarios, CUDA
agreement at ~1e-6, both-direction checkpoint interop at 1.19e-7, and two batching-invariance
tests whose teeth were confirmed by a mutation check (making `polar_express` batch-coupled moves
the result 2.41e-3). `clip_grad_norm_` at 0.4-0.5 ms and `zero_grad` at 0.003 ms are not worth
touching at 342 calls.

Two follow-ups from `modded-nanogpt` are identified but unmeasured here. Its Polar Express is
`torch.compile(dynamic=False, fullgraph=True)` and orthogonalises in bf16; ours deliberately stays
fp32 for reproducibility (documented in `optim.py`), but is not compiled. And it groups parameters
into padded *banks* rather than exact shapes -- padding ours to a multiple of 16 would take 19
groups to 9 for 2% more elements, and zero-padding is exact for the polar factor because the odd
polynomial preserves the zero block and the Frobenius norm is unchanged.

## Rejected or already addressed candidates

- **Larger PPO minibatches:** `artifacts/sweeps/update_batch_final.json` records 2,048 as the
  throughput optimum; 8,192 and 16,384 OOM. Device work per row, not launch count, is the
  constraint. Note that production does not take that optimum: `production_ppo_config`
  (`src/kaggriculture/production.py:109-123`) deliberately runs 4,096, accepting the measured
  2.2-4.8% slowdown so that 230,080 states divide into 57 balanced minibatches and the
  two-actor/four-critic schedule reproduces the former step counts while replaying each state
  more often. The lever is therefore already spent, not available.
- **CUDA graphs for the update:** the batch sweep removes roughly 97.5% of launch API calls with
  essentially no wall-clock improvement. The update is device-work-bound.
- **`reduce-overhead` rollout:** retired, not merely unprofitable. Both Inductor modes originally
  *hung* before producing an iteration -- compile workers at 0% CPU with no codegen for minutes,
  frozen allocator memory, 54 of 58 threads in `futex_wait` -- on the shipped tree and on the
  frozen baseline `a39b32c6` alike, so it was pre-existing rather than a regression from the
  attention or optimizer work. Two separate causes were found, in this order:
    1. *Lazy compilation entered from both shard threads at once.* An earlier revision of this
       entry blamed CUDA-graph capture; that was measured and is wrong, because `inductor_default`
       (`mode="default"`, no capture anywhere) hung identically. What the modes shared is that
       `torch.compile` returns a *lazy* wrapper, so the first *call* compiles, and both first calls
       came from inside the collector's two-worker `ThreadPoolExecutor`. Driving one shard's first
       step through before the peer's fixes it, and `inductor_default` now completes a wave. This
       is the same move the modded-nanogpt speedrun makes for every shape transition it knows is
       coming: execute the transition step off the clock during warmup.
    2. *A per-thread manager keyed off a process-global counter.* Only capture has this one; see
       finding 8 for the source lines. `_cuda_graph_generation` fixes it.
  After both fixes `reduce-overhead` still wedges, and two attempts to capture a stack cost
  twenty-five minutes of idle exclusive GPU between them and returned nothing. Further debugging
  of that layer is not worth the machine time when an explicit `torch.cuda.CUDAGraph` sidesteps it
  entirely -- see finding 8. `inductor_default` remains available and measured; it is fusion
  without capture, which does not address the launch count that finding 8 identifies as the cost.
- **Probes that hold the queue:** two of the above stalls were charged to a *single-slot* GPU queue
  with real training behind them. Any probe of a mode that might hang now arms
  `faulthandler.dump_traceback_later(..., exit=True)` around each wave, so it aborts in minutes
  with every thread's Python stack rather than sitting on the lease until its time limit. Attaching
  a sampling profiler after the fact needs elevated permissions this environment does not grant,
  so arming the dump in advance is the only way to get that stack at all. Probes also queue behind
  user work with `--after-terminal` rather than jumping it on priority.
- **Raw simulator parallelism:** the hot binding already releases the GIL and uses Rayon for
  per-row sampling and per-game stepping (`rust/kagg_env/src/python.rs:478-565`). An unarchived
  local scalar spot measurement was fast; persist a matched stage benchmark before reprioritizing.
- **Per-step critic inference:** behavior values are already replayed in large batches before the
  update (`src/kaggriculture/ppo.py:877-933,2072-2085`). Restoring 719 small critic calls would be
  a regression.
- **Parity audit cadence:** the source records roughly 5% on an audited iteration at a cadence of
  25, about 0.2% amortized (`scripts/train_ppo.py:589-596,717-730`). Keep the correctness gate.
- **Checkpoint I/O as a primary local bottleneck:** current single/population checkpoints are
  approximately 23.3/93.0 MB, and serialization is already backgrounded. Double serialization at
  numbered checkpoints is worth measuring only for slow/network storage; no persisted timing
  probe currently establishes it as a local bottleneck.
- **Telemetry/provenance hashing:** TensorBoard mirroring and journal hashing are incremental;
  provenance hashing is launch-time work. Neither is inside the environment or minibatch loop.

## Recommended implementation order

1. Run the critic-epoch longitudinal A/B. This has the largest measured whole-iteration payoff and
   may improve held-out loss.
2. In parallel by workflow, prototype batched/native checkpoint evaluation and the BC encoded
   cache. Both remove repeated work without changing PPO semantics.
3. Add population staging instrumentation and an iteration-scoped staged rollout.
4. Add row-role stage timing, then remove built-in/fake-ensemble work.
5. Implement direct mask bit-packing if memory headroom or population scaling matters; retain it
   only if end-to-end update time does not regress.
6. Apply the duplicate-score fix and shared-encoding microbenchmarks; consider subwave overlap only
   after the new mixed-wave profile.
7. Explore a smaller critic after the one-epoch baseline is established, so architecture gains are
   not confounded with redundant epochs.
8. Revisit backend autotuning only for long uninterrupted runs and only with compile time included.

## Benchmark protocol for all accepted changes

- Freeze one complete Python/Rust/build input tree and record its source digest.
- Use the production 112 self-play + 96 league shape and current precision.
- Use six repeats; discard cold compile separately and report both cold and steady totals.
- Synchronize the device at phase boundaries.
- Report median and individual repeats, peak host/CUDA memory, and compile time.
- Change one performance variable per matched report.
- Record host load alongside every repeat, and treat an arm measured under different contention
  than its control as uncomparable. The eager rollout is launch-bound, not device-bound: with the
  GPU at 59% utilization and no throttling, unrelated host load stretched one arm's rollout from
  20.550 s to 43.381 s across consecutive repeats while its compiled, device-bound update stayed
  inside 2 s. A control measured on a quiet machine is not a valid control for an arm measured on
  a busy one. Chain matched arms with `mlq submit --after-terminal` so the pair runs adjacent in
  time and shares whatever contention exists.
- Do not profile a full wave to answer a question wall time answers. A 720-step production wave
  produces about 1.2 million CUDA events, and aggregating them has been OOM-killed twice at 13-20
  GB RSS on a workstation shared with other jobs -- once reparsing a 2.1 GB chrome trace, once in
  `key_averages()` alone. Kernel attribution is worth that cost when the question is *where* the
  time goes, and never when the question is *how much*. In `probe_rollout_utilization.py` it is
  opt-in behind `--profile`, and the attribution keys are absent rather than zeroed when it does
  not run, so a busy fraction of 0.0 cannot be misread as a measurement.
- Preserve native/official parity, replay-parity bounds, deterministic fixed-seed actions, and
  checkpoint resume equivalence.
- For learning changes, report time-to-score over multiple seeds. Throughput alone is insufficient.
