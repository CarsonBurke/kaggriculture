# Behavior-Cloning Warm-Start from the Public v27 Agent

Status: design (no implementation yet). Owner: training pipeline.

## Why

Self-play PPO from scratch has converged to a local optimum around 34-36k
money that exploits none of the game's compounding subsystems. Measured
evidence (iteration ~95-346 of run `vapo-lv2-20260813`):

- Land purchases: exactly 0 across every audited iteration.
- Animal economy (PLACE/FEED/CARE): 0 actions, ever.
- DIG: extinct; weeds sterilize ~26% of the farm by late game.
- Hiring: peaks at ~6.7 of 16 slots; deterministic decode hires 3.
- The public v27 agent banks ~145k and beat our iteration-95 checkpoint
  0-64. The ~9x gap lives in subsystems that are mutually gating (land ->
  more tiles -> more units -> animal loops), which random exploration
  cannot cross because each subsystem alone is a shaped-reward dip.

Behavior cloning from a strong public agent moves the policy across that
gap in one step, then RL fine-tuning optimizes from a basin where the
subsystems already pay for themselves.

## Teacher: what v27 actually is

`/var/tmp/kaggriculture-kaito-v27-main.py` (sha f48c2116...) is a ~99.3%
baked 719-step action trace (`_LEGACY_ACTIONS`, zlib+base85) with two thin
runtime behaviors: actor-local WEED repair and SELL-slot ordering by price
impact and Town demand. Both seats play one coherent route (HIRE4 opening).

Consequences we accept, not work around:

- The demonstrations are nearly open-loop. BC will substantially key on
  the step-phase features (`global_features` carries step/EPISODE_STEPS
  and its complement), learning a time-indexed program with weak state
  conditioning. That is acceptable for a warm start: the goal is to seed
  subsystem usage, and RL fine-tuning supplies the state-conditioned
  corrections afterward.
- Observation diversity still exists across episodes: market prices are
  shared state moved by both players' trades, and the opponent half of the
  board varies. Extracting against a diverse opponent pool exercises that.

## Dataset extraction

New script `scripts/extract_bc_dataset.py` (CPU only, no mlq needed):

1. Run official `kaggle_environments` episodes with v27 in the recorded
   seat against an opponent pool: v27 mirror, our latest checkpoint
   (sampled and deterministic), and the starter bot. Both seats of the
   v27-mirror games are demonstrations, doubling yield.
2. Per step, record the raw observation dict for the demonstrating seat
   plus the opposing seat's private state (our training-time encode takes
   `opponent_private`; the submission path tolerates its absence).
3. Parse the engine action dict `{"farmer": cmd, "hands": [cmds],
   "market": [orders]}` back into factor targets — the exact inverse of
   `actions.compile_action`:
   - unit commands -> `UnitAction` ids via the `_MOVE_COMMAND`,
     `_PLANT_CROP`, `_PICKUP_SPEC`, `_PLACE_ANIMAL` tables' inverses;
   - market orders -> (`MarketKind`, quantity index). `QUANTITY_BINS` is
     the exact range 1..100, and a decode of v27's trace confirms its
     orders max out at quantity 53 across at most 10 slots (our
     `MAX_MARKET_ORDERS`), so every demonstrated order is losslessly
     representable; any future teacher exceeding a cap is a hard
     extraction error, not a silent clamp.
4. Compute masks exactly as `policy.act_batch` does, teacher-forced: unit
   k's mask is conditioned on the demonstration's actions for units 0..k-1
   (via `apply_unit_shed_effect`/`apply_unit_tile_effect` and seed
   decrements), and market slot masks evolve through `MarketLedger` with
   the demonstrated orders. A demonstrated action that its own mask
   forbids is a hard error — it means our mask model diverges from the
   engine, which we must know before training on the data.
5. Round-trip validation: re-`compile_action` the parsed factors and
   require exact equality with v27's original action dict for every step.
6. Output: one compressed `.npz` per episode under
   `runs/bc-dataset-<date>/` — encoded tensors (board, global, units,
   positions), factor targets, masks, active flags — plus a `manifest.json`
   with teacher sha, opponent, seed, and final banks.

Scale: 719 steps x ~1.4k decision rows per episode; 128-256 episodes gives
roughly 200-400k state rows, ample for this model size (a few M params).

## Loss and training

New script `scripts/train_bc.py`, queued through `mlq`:

- Forward the standard `FarmActor`; compute per-component masked
  log-probabilities with the existing `component_selected_logprobs`
  machinery (same masking semantics as PPO replay — no new code path).
- Loss: negative mean log-likelihood over active components (unit
  components where `unit_active`, market kind slots reached, quantity
  slots for quantified kinds), i.e. exactly the factored likelihood PPO
  optimizes, with demonstration actions substituted. AdamW, cosine decay,
  early stop on held-out episode NLL (hold out whole episodes, not steps —
  steps within an episode are nearly duplicates across the dataset).
- Metrics per head: masked NLL and top-1 accuracy for unit actions,
  market kinds, quantities; entropy of the cloned policy.
- Acceptance gate (before any RL): the BC actor loaded into
  `CheckpointAgent` and replayed on the official engine must itself buy
  land, hire 4, run animals, and bank a substantial multiple of 36k in
  self-play — target within ~2x of v27's ~145k. If greedy decode of the
  clone cannot reproduce the route, the clone failed and RL will not fix
  it.

## RL fine-tune integration

1. Initialize a fresh run whose actor weights come from the BC
   checkpoint; the critic starts fresh (BC gives no value function).
2. Critic-first fitting: for the first ~10-20 iterations run rollouts and
   critic-only updates so the actor is not pushed by advantages from an
   unfit critic. This reuses the `critic_epochs` split in
   `update_ppo` — one small extension (allow actor participation in zero
   epochs for a configured warmup) rather than a new code path. The
   existing `target_kl` gate then governs the transition, no LR hacks.
3. League: seed the frozen-opponent archive with the BC snapshot so the
   learner must keep beating its own starting point; PFSP retires it
   automatically once fully beaten.
4. Watch entropy explicitly: the clone will start far sharper than a
   scratch policy (entropy well below the ~0.66-0.80 band we see now).
   That is expected; intervene only if the in-loop external eval (see
   metrics task) shows regression toward pre-BC behavior, and then by
   diagnosis, not a reflexive entropy floor.
5. Provenance: the BC dataset manifest (teacher sha, episode seeds) and
   the BC checkpoint digest go into `run_provenance` so any resulting
   submission can be traced to its imitation source.

## Risks

- Mask-model divergence from the real engine surfaces as extraction
  errors (step 4/5 above). This is the highest-value failure mode: it
  would also mean our RL masks are wrong.
- The clone may be brittle off-distribution (opponent plays into board
  states v27 never saw). Mitigated by extraction-vs-our-checkpoint games
  and by RL fine-tuning itself.
- MAX_UNITS: v27's HIRE4 route stays far below our rust cap of 16, so the
  known latent cap divergence stays latent.
- Kaggle legality: v27 is a public notebook agent; training on its public
  trace is standard practice, but the submission must be our network, not
  the trace — which this plan guarantees by construction.

## Effort estimate

- Extraction script + inverse-action parser + round-trip tests: ~1 day.
- BC trainer + metrics + acceptance replay: ~0.5-1 day (GPU hours: <1 on
  the 5090 at this model size).
- RL integration (critic-first warmup flag, league seeding, provenance):
  ~0.5 day.
