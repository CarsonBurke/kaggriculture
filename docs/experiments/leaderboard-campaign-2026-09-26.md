# Leaderboard campaign — 2026-09-26

## Delivered result

Strongest tested policy submitted as **56589679**: a reviewed, isolated
three-to-four-turn sale-lookahead change to tetsutani's public demand policy.
Its active companion is the unchanged parent **56589389**. Both are COMPLETE
on Kaggle. The new variant's hosted validation replay (113868573) was downloaded
and checked: environment 1.32.7, 720 states, no historical errors, terminal banks
matching rewards. Public attribution and Apache-2.0 notices are preserved.

At the 22:00:27 UTC leaderboard snapshot, the active best was **898.7, rank
4454/10057**, versus **810.7, rank 4819/10054** before this campaign. The newest
variant starts at rating 600 and has not yet established a public ladder record.
These early scores do not establish top-ladder strength or the eventual rating.
Two submission slots remain today. All local campaign jobs and all five current
remote notebook versions are complete; no training jobs were left running.

The improved variant's separate final panel: **50 wins, 4 draws, 10 losses**
against its parent; **58/64 wins** against hybrid versus the parent's 52/64 on
the same maps; **64/64** against idle seller. 322 full final/mirror games and
eight fresh-worker exact-bank repetitions passed. Runtime, stderr, source hash,
archive and coverage audits passed; five evaluator regression tests pass.

## Objective and plan

Optimize actual Kaggriculture leaderboard wins. Do not treat improved PPO
training curves, teacher-forced accuracy, or wins against saturated old
opponents as sufficient evidence.

1. Establish the live leaderboard, current executable public baselines, and
   submission rules.
2. Screen stronger agents on paired fresh maps and a separate current-replay
   panel. Reproduce recorded games before using their frozen action streams.
3. Test isolated, evidence-backed improvements; select by wins, with bank
   margins and behavior as diagnostics.
4. Freeze the candidate, evaluate unused maps and reserved replay episodes,
   validate the exact standalone package, then submit and inspect live games.

## Initial evidence

The account's latest neural submission is `56484045`, rated 812.9 at inspection.
The historical best listed submission is `55613235` at 1429.9. These ratings
are not stationary comparisons. The current leader was rated 3079.3.
Kaggle's evaluation page says only the latest two submissions remain active
and are used for the final leaderboard; five submissions per day are allowed.
No campaign submission had been made at baseline capture.

Immediately before finalist validation, the full downloaded leaderboard listed
our team at rank 4819/10054, score 810.7. The top score was
3077.5. The CSV snapshot is retained in `leaderboard-before/kaggriculture.zip`.

The previous action-stack runs use much older v16/v27 opponents. The strongest
well-supported local neural candidate is the mean of R1 snapshots 100–131,
with 100% development wins against starter/v27 but only 19.9% against v16.
MLQ job 10211 evaluated it against current public Kaito V48: one exclusive
GPU slot, default priority, 30-minute cap, full games with compiled BF16 inference.
It won 4/32 games (16 paired maps), with mean final bank 29,363 and mean
margin -104,851. All 32 games were valid and complete. The failed first attempt
was an export argument error (population agent=0 on a single-learner checkpoint),
corrected to agent=None before the successful run. No model behavior was changed.
Evidence: `artifacts/probes/leaderboard-20260926/r1-vs-v48.json`.

Public source and all campaign artifacts are under
`artifacts/probes/leaderboard-20260926/`. Public sources retain their original
contents and metadata; candidate files are separate.

## Evaluation design

- Official `kaggle-environments==1.32.7`, matching the downloaded live replay.
- Initial public-agent round robin: V48, V20, v16, v27; 16 fresh maps starting
  at 10426000, both seats, 192 games.
- Development replay panel: four recent episodes from current strong teams,
  eight action streams, both counterfactual seats. A newer disjoint episode
  per source team is reserved for final evaluation. Duplicate episodes are
  excluded across the development/finalist split.
- All four recorded development games reproduce their exact final balances.
  V48 wins 12/16 counterfactual games; V20 wins 6/16. Frozen opponents cannot
  react to changed play, so these are screening results, not leaderboard
  predictions. There are only four independent map clusters in this panel.
- V20 ablation maps start at 10426100; V48 ablation maps at 10426200.
  The final map panel remains unused.
- Every full game must have 720 states, valid dictionary actions, no historical
  ERROR/INVALID/TIMEOUT states, finite banks, and matching terminal rewards.
  Final DONE status alone is insufficient because the interpreter overwrites
error status at the last turn. Five focused evaluator regression tests pass.

Scripted-agent tournaments run in private Kaggle CPU notebooks while unrelated
work occupies the local MLQ queue. No paid compute is used. The initial reference
uses one fresh process per game. Later batches use persistent workers;
the official loader executes each agent source into a fresh namespace per game.
The attempted 16-game worker recycling deadlocked under Python 3.12 after
32 results, consistent with CPython issue 115634. Those incomplete batches
are not selection evidence. The runner now allows only fully isolated or
persistent workers, with explicit collection of discarded environment cycles.
The exact source, task lists, and worker setting are retained with each report.

## Candidate experiments

V20 candidates test separately:

- Correct carrot/tomato/egg scarcity curves to the current hinge functions.
  36,009 scalar price checks agree exactly with the official implementation.
- Sell all actual final-turn stock, including same-turn deposits, without
  duplicate orders, preserving original product priority.
- Rank sale slots after counters and guards add or enlarge orders.
- Combine those changes only as an explicitly separate candidate.

V48 candidates test separately:

- Require a current public-farm match before continuing clone-sale preemption.
- Recover carried goods at turns 717–718 when they can still reach the shed,
  then recompute final liquidation.
- Combine those changes as a separate candidate.

The first variant ablation was invalidated by callable discovery: rebinding an
existing Python entrypoint does not move its position in the namespace, while
Kaggle chooses the last inserted callable. Every candidate now has a uniquely
named final entrypoint, checked using the official discovery rule. Failed v1
variant runs are not performance evidence. Corrected v2 batches hit the worker
recycling issue above; v3 uses persistent workers.

Each of the following ran in its own private Kaggle notebook:

- the current-agent tournament
- the current replay panel
- the market ablation
- the V48 ablation

The current notebook catalog also contains agents published more recently than
V48/V20. Their source is being reviewed before expanding the benchmark; notebook
titles and self-reported scores are not treated as verified performance.

## Completed initial tournament

Across 16 fresh maps and both seats, V48 won 28/32 against v16, 30/32 against
V20, and 31/32 against v27. V20 versus v16 was 16/32. These results demonstrate
why older opponents are insufficient for selecting a leaderboard candidate.
Raw results: `artifacts/probes/leaderboard-20260926/tournament-output/tournament.json`.

## Current frontier screen

Three unique newer public agents are evaluated against V48/V20, each other,
and the development replay panel: haideptry's hybrid engine, tetsutani's
demand-preserving sale timing, and lynnsakurai's idle seller. The apparent
fourth candidate (flexonafft multi-route) duplicates the demand agent after
line-ending normalization and is excluded. Source hashes were verified against
the public notebooks' actual output files. The private evaluation notebook
uses those public outputs as inputs and rejects any source-hash mismatch.
It is a separate private Kaggle notebook (192 games).
The optional idle-seller external library environment variable is unset.

## Hybrid bookkeeping ablation and frozen finalists

The hybrid records planned own sales before later wrappers advance orders.
An isolated candidate reconciled that snapshot with the final returned action.
Pure-data checks reproduced and corrected false opponent-sale attribution, but
MLQ 10214's 160 full games did not show a win improvement: the patch scored
40.6% against its parent, 25% against demand timing, and 87.5% against idle
seller. The parent scored 25% against demand timing and 93.75% against idle.
The change is rejected for submission; correctness of an intermediate estimate
alone is insufficient evidence of a stronger policy.

Demand timing beat the original hybrid in 24/32 paired games on those 16 fresh
maps. Both original sources are now frozen into deterministic archives under
`artifacts/probes/leaderboard-20260926/finalists/`. Their actual source headers
contain Apache-2.0 notices and the full license; these are preserved, with
public-notebook provenance in an added NOTICE. No policy code was modified.

MLQ 10215 evaluates 320 responsive games across 32
unused maps beginning at 10427000, four full mirror games, four exact reserved
replay reproductions, and 32 frozen-opponent counterfactual games. The responsive
panel contains both finalists versus each other, idle seller, and V48, in both
seats. One exclusive queue slot, four CPU workers, default priority, 40-minute
cap. The evaluator is snapshotted for this job and records action timing and
stderr alongside historical failure checks. Finalist source hashes and archive
hashes are recorded before evaluation and source hashes checked after it.

## Final validation and submissions

All 368 final/audit games completed: 324 responsive/mirror, 36 reserved replay
including four exact original reproductions, and eight fresh-worker repetitions
in reversed order. Fresh workers reproduced exact terminal banks. No historical
agent failures or stderr were recorded. Maximum local action time was 0.254s.
Report source hashes, environment 1.32.7, expected task coverage, and archive
hashes passed the final audit. An independent reviewer also checked archives
and task coverage. These are local timing measurements, not hosted guarantees.

On 32 unused maps, both seats (64 games per opponent):

| Candidate | Hybrid | Idle seller | V48 |
| --- | ---: | ---: | ---: |
| Demand timing | 50/64 | 64/64 | 64/64 |
| Hybrid | — | 55/64 | 64/64 |

Reserved frozen-replay wins: demand 10/16, hybrid 8/16. Those results have only
four independent map clusters and nonreactive opponents; they are not rating
predictions. The demand policy is the primary candidate; hybrid provides the
second validated active policy.

Submitted on 2026-09-26 UTC, with upstream public authors credited:

- Hybrid `56589375`, archive SHA256
  `de6e8176f3bf8ac9eb0e346ed7006f17ffeecd725310c670ebfd4291d258a68a`.
- Demand timing `56589389`, archive SHA256
  `8cfb8b13d2f9adbadaa96d1731af975b73ab8d197286eb2a4107f08f0c2d61cd`.

Both uploads succeeded; hosted validation and rating are pending at this entry.
Three submissions remain today. Do not resubmit identical archives merely
because rating or validation is pending.

The completed V48 ablation found all three variants behaviorally identical in
terminal banks to their parent on this opponent panel (mirror score 50%, margin
zero; same 27/32 wins versus V20). None is selected for submission.

The completed 192-game frontier screen confirms both finalists won all 16 games
against each older responsive baseline (V20 and V48). Demand beat hybrid 7/16
on this eight-map screen, versus 24/32 in the separate development ablation and
50/64 on the unused final panel. Development frozen-replay results differed
(demand 7/16, hybrid 11/16), reinforcing that frozen tapes are not a substitute
for responsive games or hosted rating.

At 21:38 UTC, hybrid passed hosted validation episode 113862367 and entered
matchmaking at the default rating 600. This initial value is not performance
evidence. Demand remained in hosted validation at that check.

Demand subsequently passed hosted validation episode 113862371 as well. Both
submissions are COMPLETE and initially rated 600. The downloaded hybrid hosted
replay uses environment 1.32.7, has all 720 states, no historical error statuses,
and two DONE players. Hosted validation archives are retained compressed.

While hosted ratings develop, MLQ 10218 tests one further isolated development
candidate: demand policy sale lookahead 3 -> 4, borrowing the hybrid's existing
setting while preserving all other demand policy layers. 160 full paired games
on new maps beginning at 10428000; one exclusive slot, four CPU workers, default
priority, 30-minute cap. This is a fresh development experiment, not a change to
either submitted archive. Longer anticipation could improve sale priority or
damage demand recovery; promotion requires win evidence and new final maps.

The four-turn candidate won 27/32 against its parent and matched its external
opponent win totals (30/32 hybrid, 32/32 idle seller). Small bank margins did not
improve against those external opponents, so no broad strength claim follows.
MLQ 10219 now tests 32 new final maps starting 10429000: 320 responsive games,
two mirror games, and eight exact-bank fresh-worker repetitions. The source is
frozen at SHA256 `d1286a994ed6df6c5e4c54d0e8c8afa2ee3c447bba569d47ddb4d8a7701db33e`.
Its archive retains the original license and states the specific modification.
Neither existing submission was changed. Exclusive slot, four CPU workers
(two for isolated repetitions), default priority, 40-minute cap.

Completed V20 ablation: sale ranking and the combined patch each beat their
parent 32/32; corrected curves scored 30/32, terminal handling 24/32. However,
every V20 variant remained at 4/32 versus V48, exactly the original's win total.
These are real narrow improvements, not evidence of a competitive replacement
for the frontier agents. No V20 variant is submitted.

First verified public ladder result: demand submission 56589389 won episode
113863774, 193,287 vs another team's 44,477, with 720 states, environment 1.32.7,
and no error statuses. Its rating rose from 600 to 668.6. This is an initial
low-rated match, not evidence of top-ladder strength. The replay is compressed
under `hosted-review/`; subsequent rating development is still being observed.

The four-turn candidate's separate 32-map final panel completed successfully.
Against its parent it had 50 wins, 4 draws, 10 losses (81.25% score); versus
hybrid it won 58/64, compared with the parent's 52/64 on those same maps;
both won 64/64 against idle seller. The parent comparison is **not** 52 wins:
52 is the equivalent score after half-credit for four draws. The evaluator now
reports explicit wins/draws/losses to prevent conflating score with win count.
All 322 final/mirror games and eight exact-bank fresh-worker repetitions passed,
with no stderr and maximum local action time 0.277s. Frozen archive SHA256:
`3f59f2a9784c3e0bd39410742c63066e5dae63a502c187d510ef6da5acafa3f7`.
The source delta is isolated, reviewed, and retains a unique final callable;
rebinding inherited global `agent` is deliberately avoided because it can
create recursion through the upstream wrappers.

After independent final promotion review, submitted the four-turn variant as
`56589679` on 2026-09-26 around 21:55 UTC. Upload succeeded; two submissions
remain today. The latest pair becomes demand timing `56589389` and this improved
variant, retiring the older hybrid under the latest-two rule. At upload, the
parent was rated 808.0 after three public episodes; hybrid had reached 866.3
after three episodes, exceeding the original active baseline 810.7. These early
ratings are not used to override the matched final-panel evidence; ratings
are still developing from their default 600 starts.
