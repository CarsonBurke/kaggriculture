# Structured VIT Improvement and NextLat Proposal

## Decision

Keep the structured VIT as the actor baseline. Improve it along three independent axes:

1. **Typed global memory with repeated cross-attention.** Products, crops, farms, town/clock, units, and opponent summaries remain entity tokens; cross-attention is how central latents read those entities. “Entity tokens or cross-attention” is therefore a false choice.
2. **A small, controlled subset of modded-nanogpt residual and initialization ideas.** Start with static input reinjection and a single global refresh. Add dynamic MUDD routing only after the static paths prove useful.
3. **Structured NextLat with future 1x1 tile-patch prediction.** Preserve the already useful decision-decode KL, but make the spatial target the post-farm-local patch representation and condition the predictor on factored action tokens rather than one globally summed action vector.

Do not import diffusion sampling, full MUDD, XSA, FP8, or custom Triton kernels as a bundle. They solve different bottlenecks and would make attribution impossible.

`RUNS.md` is the execution ledger. Every learning change is a champion-versus-challenger regression. Exact systems changes may be adopted after numerical parity and end-to-end timing; learning changes require complete BC and play evidence before PPO promotion.

## Current model: what global fusion already does

`StructuredTrunk` already treats the non-spatial state as typed entities:

- one token per product;
- one token per crop/seed family;
- two farm-summary tokens;
- one town token;
- up to 16 own-unit tokens;
- eight opponent-summary tokens produced by cross-attending learned queries to the opponent’s 100 patches.

The 32 central latents cross-attend this context once in `latent_read`. The market decoder later cross-attends the economy tokens directly. Unit outputs receive a direct five-patch HERE/N/S/E/W path.

The missing capability is not initial global access. It is **refresh**: after `latent_read`, eight self-attention core blocks must preserve every economic and opponent fact without reading the typed context again.

## Proposed global-memory design

### Keep global values as entities

Do not collapse products, crops, farms, or town state into one pooled vector. Their identities align with market actions and with each other. A product price, shed quantity, market quantity, and product-specific order should remain addressable through the same product token.

Split the current town token into a town token and a clock/phase token only as an isolated tokenization regression. Do not bundle that change with repeated cross-attention.

### Add a mid-core global refresh

The first implementation adds one `GlobalRefresh` after core layer 4:

```text
initial_context = [own patches, opponent summary, units, economy]
latents = latent_read(latent_queries, initial_context)
latents = core[0:4](latents)
latents = global_refresh(latents, global_memory)
latents = core[4:8](latents)
```

The first `global_memory` arm contains only economy, farm, town, and clock entities. The competing arm adds own units and opponent summaries. Raw own and opponent patches are excluded: the initial read and direct local decoder already carry spatial information, and rereading 200 patches would pay the largest cost for the least targeted hypothesis.

The refresh uses pre-RMSNorm, QK-Norm, SDPA, and the existing gated residual/FFN block. Its residual gate initializes at zero so the candidate starts functionally at the champion while the gate can learn whether the path is useful. If two refresh sites win later, normalize and project global keys/values once per forward and reuse them at both sites.

This is the structured analog of modded-nanogpt value-embedding reinjection: the model can recover stable input information at depth, but cross-attention preserves the fact that global memory and latent workspace have different token identities and lengths.

### Cheap conditioning alternative

DiT-style adaptive normalization is a secondary regression, not the default. A pooled clock/farm vector may modulate residual gates or RMSNorm scale in each core block, but it cannot replace entity-addressable product and crop memory. Test it only after the cross-attention arm establishes whether repeated conditioning helps.

## Modded-nanogpt transfers

### Already adopted

The current stack already contains the transferable baseline:

- RMSNorm and QK-Norm;
- 2D RoPE;
- ReLU-squared FFNs;
- bias-free attention projections;
- gated residual paths;
- NorMuon, Polar Express, low-rank second-moment correction, and cautious weight decay;
- BF16 activation flow and compiled rollout/update paths.

These are not new regression arms.

### Implement now

#### 1. Exact graph and data-path reductions

These should preserve the represented function and need timing rather than retraining evidence:

- Encode own and opponent farms as one `[2B, 100, D]` batch through the shared farm-local blocks instead of invoking the same blocks twice.
- Use the canonical board ordering to expand the precomputed RoPE table directly instead of gathering the same 100 positions every forward.
- Fuse same-width categorical embedding tables with fixed index offsets so tile and unit categories use one embedding lookup each.
- Cache immutable quantity-head arrays in prepared inference agents.
- Batch official evaluation environments in lockstep around one accelerator actor.
- Cache architecture/provenance-bound BC encodings so each regression does not repeat JSON decode and structured tokenization.

Every change requires eager/compiled output parity, unchanged legal actions under fixed RNG, backward finiteness, and batch-one/rollout/update benchmarks.

#### 2. Static latent-input reinjection

Cache the output of `latent_read` as `x0`. Add a learned per-channel reinjection at core layers 3 and 6:

```text
x = x + gate[layer] * RMSNorm(x0)
```

Begin with static learned gates. This is the boring version of modded-nanogpt embedding skips and MUDD residual routes. It costs one normalized add, preserves token alignment, and answers whether the core loses early information before dynamic routing is justified.

A separate arm may add one U-shaped skip from the layer-2 snapshot into layer 6. Do not combine both in the first test.

#### 3. Zero-initialized branch outputs

Test zero initialization of attention output and FFN down projections as an alternative to the current random branch plus 0.1 LayerScale-like gate. Do not zero the branch projection and its residual gate simultaneously; that suppresses useful gradients. The regression must compare complete 12-epoch convergence because the current short BC schedule may favor the existing nonzero path.

#### 4. Fused self-attention projection, only with optimizer semantics preserved

A fused QKV projection can reduce self-attention launches, but placing QKV in one parameter changes NorMuon’s matrix geometry. It is not an exact systems change unless the optimizer still orthogonalizes the intended Q/K/V or per-head slices independently. Profile the projection share first. If material, implement a contiguous parameter bank with explicit optimizer slices and run it as a learning regression.

### Implement only after simpler arms win

#### MUDD-lite

Use one dynamic routing site near the final core block. It may mix `{x0, layer2, layer5, current}` with coefficients predicted from the current normalized latent. Initialize its effective coefficients to reproduce the static champion exactly. Promote only if static reinjection already helped and the dynamic route improves more than its latency and complexity cost.

#### Layerwise global key/value reuse

If two global refreshes beat one, share the normalized global K/V projection across sites and retain layer-specific query/output projections and gates. This is a more faithful structured analog of value embeddings than copying one pooled vector into every latent.

### Defer or reject

- **Full MUDD across all blocks:** too much routing complexity before one useful skip is established.
- **XSA and paired-head attention:** designed around redundant long causal language attention; no evidence that 32 noncausal latents have the same failure.
- **Long/short sliding windows and partial key offsets:** the board uses exact 2D topology and only 100 patches.
- **Bigram embeddings, smear, prefix-token prediction:** token-sequence-specific. Their relevant analog here is action-conditioned future-state prediction, covered by structured NextLat.
- **FP8 and custom Triton FFN kernels:** the model consists of many small operations and is not yet shown to be GEMM-throughput bound. Remove launches and blocks first.
- **REPA-style external alignment:** no stronger external representation encoder currently exists. Self-alignment to an arbitrary latent can optimize predictability rather than play.

## Structured NextLat

### Goal

Train the structured actor to preserve representations that predict both:

1. what the policy will need to decide at the next step; and
2. how each 1x1 board patch will evolve under the executed action.

This combines the decision-decode result already measured in `NEXTLAT_AUX_PLAN.md` with the spatial feature-prediction principle demonstrated by V-JEPA and the action-conditioned world-model direction demonstrated by V-JEPA 2-AC.

It remains an auxiliary representation objective. The actor still produces exact one-pass masked categorical policies; no world-model rollout enters deployment unless a later planning experiment explicitly proves useful.

### Prediction sites

Add a typed `StructuredBelief` return path without changing `StructuredActor.forward`:

```text
StructuredBelief
  own_patches       [B, 100, D]
  opponent_summary  [B, 8, D]
  economy_entities  [B, E, D]
  central_latents   [B, 32, D]
  unit_decisions    [B, 16, D]
  market_decisions  [B, 10, D]
```

The primary future-patch target is `own_patches` after the shared farm-local blocks and before `latent_read`.

This site is deliberate:

- it preserves one token per actionable board tile;
- it contains local relational context rather than only raw feature embeddings;
- it is consumed by both the latent read and the direct unit-local path;
- it avoids asking 32 central latents to reconstruct where spatial facts came from.

Do not use tokenizer outputs as the primary target; copying mostly unchanged input embeddings is too easy. Do not use only final policy tokens; the existing entity-CNN experiment already did that and omitted patch dynamics.

### Factored action entities

Replace the previous single summed joint-action embedding with typed action tokens:

- 16 unit-action tokens carrying unit slot, opcode/direction/object identity, current position, and active state;
- 10 market-action tokens carrying order slot, kind, product/crop identity, and selected quantity;
- learned unit-action and market-action type embeddings.

The environment action is a structured set, not a bag. Summing every action before prediction erases which unit changed which patch and which market order changed which product.

### Predictor

Use one narrow training-only transition block:

1. Form prediction queries from the current typed belief tokens plus token-type/position identity.
2. Form transition context from current central latents and factored action entities.
3. Cross-attend every prediction query to the transition context.
4. Apply a shared ReLU-squared FFN.
5. Predict a residual delta for each query token.

For horizon greater than one, feed predicted central latents and typed tokens back into the predictor with the next demonstrated action entities. Keep target types separate in the loss even if the predictor block is shared.

Expose separate `decision_horizon` and `patch_horizon` settings. Decision KL starts
from its measured horizon of two; patch prediction starts at one step so a failed
spatial objective is not confounded with recursive predictor error. Remove the
single shared horizon once the structured path replaces the entity-CNN-only
implementation.

Central latents are the predictor's recurrent workspace. Do not add a raw
central-latent regression term in the first implementation: the earlier
entity-CNN evidence showed that coordinate predictability can compete with the
policy. Supervise predicted central latents indirectly through future patch and
decision losses, including their recursive-horizon gradients.

The predictor belongs to the trainer checkpoint for resume, but not to actor artifacts, frozen league actors, or submission bundles.

### Losses

#### Decision decode

Retain the proven objective:

- decode predicted `unit_decisions` and `market_decisions` through detached policy heads;
- decode the true future decision tokens through the same detached heads;
- compute legality-masked KL from true-future distributions to predicted-future distributions;
- start from coefficient 0.5 and horizon 2 because that is the measured useful setting.

#### Future own-patch features

Predict the next own-farm post-local patch features with stop-gradient targets. Use feature-space L1 after parameter-free RMS normalization, matching the robust feature-prediction form rather than raw unnormalized coordinate regression.

Most board patches often remain unchanged. Report separate losses for changed and unchanged patches and optimize:

```text
patch_loss = 0.5 * mean(all eligible patches)
           + 0.5 * mean(changed eligible patches)
```

A patch is “changed” only when its canonical staged categorical or continuous tile fields differ at the successor step. If a row has no changed patch, its second term contributes zero rather than inventing a denominator.

The patch coefficient is calibrated once on a frozen, recorded batch so its trunk-gradient norm is 10–30% of the clone-loss trunk-gradient norm, then held fixed for every matched arm. Do not use online adaptive loss balancing.

#### Other state types

Economy entities, opponent summaries, and opponent patches are separate regressions
after own-patch prediction is proven:

- Economy prediction is highly relevant but can be dominated by deterministic clock
  and price motion.
- Opponent evolution is partly uncontrolled and multimodal because the actor does
  not know the opponent’s action. Test eight summary targets before adding all 100
  post-farm-local opponent patch targets.
- Do not condition an actor auxiliary on the opponent’s executed action even when a
  self-play trajectory can expose it. That would create a privileged transition
  model unavailable to the deployed actor.

Never bundle these with the first own-patch arm. Record per-type losses and promote
each independently.

### Target encoder

Version 1 uses the same actor at the true successor observation with stop-gradient, matching NextLat and the existing implementation. BC supervision and decision KL provide anti-collapse pressure.

An EMA target trunk is a separate V-JEPA-style regression, not an unreported implementation detail. Test it only if online targets are unstable or the patch branch collapses. It adds a target forward and target parameters, so its quality gain must justify training cost.

### BC integration

BC already stages `episode_index`, `step`, and contiguous runs. Extend the structured actor’s `forward_with_belief` path and the existing `latent_horizon_loss` machinery to typed beliefs rather than flattening heterogeneous tokens into one tensor.

The no-auxiliary contiguous-sampler control remains mandatory. It isolates the sampler from the objective.

### PPO integration

Only implement PPO-active NextLat after BC establishes a play improvement.

Do not reshuffle PPO policy minibatches into temporal runs. For each ordinary actor minibatch, draw a separate contiguous transition minibatch for the auxiliary and combine both losses in one backward/optimizer step. Concatenate source and successor observations for one compiled structured-actor call. This preserves PPO sampling semantics while giving the auxiliary exact successors.

Compare:

1. NextLat during BC only;
2. the same pretrained actor with decision and patch losses active throughout PPO.

No automatic decay schedule in the first comparison. If PPO-active NextLat helps early and hurts late, a predeclared schedule becomes a later regression.

### Required diagnostics

For every NextLat arm record:

- decision KL by unit, market kind, and quantity;
- patch L1 for all, changed, and unchanged patches;
- one-step and horizon-k loss separately;
- patch feature variance, effective rank, cosine similarity, and mean pairwise dispersion;
- predictor residual magnitude relative to current-token magnitude;
- clone loss and policy accuracy;
- BC epoch time and PPO update time;
- off-distribution deterministic play, not only holdout NLL.

A falling patch loss with falling feature variance is collapse, not progress. A low unchanged-patch loss with poor changed-patch loss is identity copying, not a world model.

## Performance frontier after quality improvements

The structured actor is currently stronger but slower than the entity-CNN. After selecting useful representation paths, search for a cheaper equivalent rather than shrinking before the signal is understood:

1. 32 to 24 central latents.
2. Eight to six core blocks.
3. Six core blocks plus the winning global refresh, testing whether fresh context replaces depth.
4. Fuse market latent/economy decoding into one cross-attention context.
5. Profile a single combined unit context containing central latents plus the five local patches before replacing the two unit decoder blocks.

Separately reduce critic cost. The actor and critic need not share depth:

- test critic core depth 6, then 4;
- test 16 critic latents;
- retain exact centralized private fields and the distributional value head;
- test three critic epochs during warmup and one after warmup against the current fixed two.

Rank critic arms by next-wave value quality and time-to-external-score, not same-wave fit.

## Promotion policy

- Exact systems changes: numerical/action parity plus lower complete wall time.
- BC architecture changes: complete 12-epoch runs, deterministic off-distribution play, and no material throughput regression without a clear quality gain.
- NextLat: multiple BC seeds because prior results were basin-dependent; selection by play, never NLL alone.
- PPO: at least three matched seeds through the complete 100-iteration gate before continuation.
- Finalists: continue the same checkpoints to 500 iterations; do not restart a lucky curve under a different seed schedule.
- All local accelerator jobs go through `mlq`.
- No shortened 720-step games, reduced model, or partial training horizon is accepted as quality evidence.

## Research basis

- Modded-NanoGPT: <https://github.com/KellerJordan/modded-nanogpt>
- NextLat: <https://arxiv.org/abs/2511.05963>
- V-JEPA feature prediction: <https://arxiv.org/abs/2404.08471>
- V-JEPA 2 and action-conditioned world models: <https://arxiv.org/abs/2506.09985>
- Perceiver IO: <https://arxiv.org/abs/2107.14795>
- DiT adaptive conditioning: <https://arxiv.org/abs/2212.09748>
- REPA representation alignment: <https://arxiv.org/abs/2410.06940>
- ViT-5 modernization: <https://arxiv.org/abs/2602.08071>
