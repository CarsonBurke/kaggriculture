The selected working default for greedy deployment is **source-read critic + hardness league + NextLat off**, retaining the current actor GAE lambda. This is a comparative default decision: it preserves substantially more deployed strength than the other recent trained recipes without giving up much critic fitting. It is not a claim that every metric improves or that the auxiliary alone caused the difference. The new defaults are implemented in [production.py](../../src/kaggriculture/production.py).

The previous report overemphasized every candidate losing to BC. That misses the useful ranking between candidates. The cross-run evidence (`artifacts/probes/credit-valuation-20260918/cross-run-comparison.json`) now combines evaluated checkpoints, checkpoint-aligned actor updates, complete logged trajectories, common iteration windows, and the older architecture panels. [Job outcomes and checkpoint provenance](credit-valuation-results.md) remain available separately.

**Recent trained-policy ranking.** Each score below uses 256 common development maps per opponent, balanced seats, with argmax and temperature-1 sampling evaluated separately. These are scores against v27, not head-to-head matches between candidates.

| Run | Evaluated wave | Actor updates | v27 argmax | v27 sampled |
| --- | ---: | ---: | ---: | ---: |
| NextLat off, original | 35 | 609 | **94.92%** | **12.89%** |
| NextLat off, replacement | 94 | 2,320 | **91.02%** | 3.52% |
| Economic critic, NextLat on | 36 | 696 | 87.89% | 12.50% |
| Tile bias, NextLat on | 48 | 870 | 81.64% | Not evaluated |
| Lambda 1, NextLat on, original | 137 | 3,538 | 33.20% | **6.25%** |
| Hardness, NextLat on | 125 | 3,161 | 1.17% | 0.39% |
| Control, NextLat on | 120 | 2,987 | 0.78% | 0.00% |
| Source-read, NextLat on | 112 | 2,755 | 0.00% | 0.00% |
| Writeback, NextLat on | 93 | 2,146 | 0.00% | Not evaluated |
| Lambda 1, NextLat on, replacement | 72 | 1,653 | 0.00% | 0.00% |

NextLat-off wins the recent deployed-policy comparison. Its later checkpoint remains much stronger under argmax after considerably more updates than tile bias or the evaluated economic critic. Among the longer trained checkpoints with sampled evaluations, the original lambda-1 run scores better than NextLat-off: **6.25% versus 3.52%**. Therefore NextLat-off does not dominate every metric. Its original 12.89% sampled score differs from economic's 12.50% by only one game; the evidence does not establish their ordering at these early snapshots. All recent argmax starter scores are 100% except hardness at 99.61%; that saturated floor cannot rank these models usefully.

The shared BC actor scores 98.05% argmax / 26.17% sampled. Tile-bias and writeback use separate clones, starting at 96.48% and 98.44% argmax respectively. Unequal updates, changed league behavior, nondeterministic retries and interrupted runs limit causal attribution. They do not erase the practical superiority of the observed NextLat-off recipe for retaining greedy strength. Missing sampled tile/writeback results are missing evidence, not zero scores.

**Training diagnostics across the on/off recipes.** The following common window is waves 75–94, using all twenty journal rows per run. These are on-policy diagnostics on different trajectories, not fixed-data critic tests.

| Mean metric | Source-read + NextLat | Source-read + hardness, NextLat off |
| --- | ---: | ---: |
| Training money | 50,783 | 50,728 |
| Monte Carlo explained variance | 0.466 | 0.456 |
| Opening terminal explained variance | 0.117 | 0.109 |
| Value loss | 2.917 | 2.922 |
| Policy entropy | 0.162 | 0.161 |

The exact combined hardness + source-read + NextLat-on control never trained: its correctness prerequisite failed on the experimental BiXT memory test. This compares the available recipes rather than isolating the auxiliary alone. Critic fitting and the training economy are almost unchanged. The final twenty logged waves of source-read/on reach money 51,643 and MC explained variance 0.494; off reaches 54,244 and 0.468. One-step NextLat barely improves on persistence: final-twenty mean combined prediction/persistence loss ratios are 0.995 for control, 0.967 for source-read and 0.997 for hardness. Lower is better. This is a small predictive gain, not evidence of useful planning.

Production benchmarks recorded 11.55 seconds/iteration for off and 12.87 for lambda-1/on; the actor-lambda change does not change network computation. Actual training timings vary, so treat that as favorable cost evidence rather than a universal speedup. Removing NextLat also restores individual-state minibatch shuffling. The default decision adopts that whole recipe; a future attribution experiment must hold minibatch organization fixed.

Actor NextLat has stronger negative evidence in the current entity implementation. At wave 32, with thirteen actor-active waves in each arm, actor-NextLat money is 15 versus control's 53,873. Last-ten median actor gradient norm is 19.59 versus 1.36, and MC R² is −0.0034 versus 0.1572. Keep it off. The original comparison (`artifacts/probes/entity-attention-d96-20260914/actor-head-nextlat-partial-results.json`) records the full evidence; this does not imply all historical forms of NextLat failed.

**Who wins the other comparisons?** Hardness leads the recent original architecture arms on final-twenty training money: **58,127**, versus source-read 51,643, control 50,599, tile bias 46,622 and writeback 42,293. Opponent mixtures differ, so this is descriptive training productivity. Source-read wins the common-policy critic fit: held-out all-state R² **0.2144 versus 0.1165** for entity-only pooling. Its opening fit is still weak, and its final-32-step fit is worse. Retain both useful defaults without claiming they establish a gameplay win separately.

The economic critic's saved wave-36 evaluation hides a worse later trajectory. At waves 41–60, money/MC explained variance is **21,286 / 0.290**, versus NextLat-off entity **51,044 / 0.432**. Its final twenty logged waves through 69 average only **11,302** money and opening EV **0.046**. It should remain experimental. Lambda 1 is similarly inconsistent: the original run has the strongest final-twenty money among the recent credit arms, 58,368, but its replacement collapses to 0% v27 in both modes. Retain actor lambda 0.97218.

**Older runs matter, but money and winning separate.** The older GQA continuation reaches final-twenty training money **76,175** at wave 440; combined-four reaches **73,237** at waves 251–270. Their sampled fixed-panel v27 scores are respectively **0% at checkpoint 420** and **0.39% at checkpoint 271**, despite v27 money of 70,905 and 70,237. The preceding piecewise control scores 0.39%; untied K/V, intermediate FFN, local readout, critic FFN and tile RoPE each score 0%. These panels use a different development seed block, so compare descriptively, not as paired games. The older strong-money models do not supply a stronger v27 sampled result than the recent off recipe.

An older structured balanced run reaches training money **76,721** and MC EV **0.878** through wave 411. Its reward and learning setup differ, so that EV is not comparable to current terminal-outcome EV. It nevertheless argues against dismissing structured/global reasoning or temporal auxiliaries in principle. The GQA continuation and combined-four also improve well after early money plateaus; thirty stale money/value-loss waves are not a reliable architectural rejection criterion.

**Defaults chosen and applied:** hardness selection with discovery/stale refresh; source-read entity critic; actor and critic NextLat coefficients zero; current actor lambda; existing efficient entity actor. Economic critic, writeback, tile bias and BiXT stay opt-in. Historical campaign builders explicitly pin their original NextLat-on controls, including benchmark coefficients, so the new default cannot silently rename their experiments. Existing checkpoints are unchanged. Resume validation rejects a configuration mismatch; continue old NextLat-on runs with their frozen source and original configuration rather than applying the new defaults.

The next experiments are the [selected structural ablations](architecture-ablations-2026-09-18.md). Better diagnostics accompany them; they are not an indefinite prerequisite to ambitious architecture work.
