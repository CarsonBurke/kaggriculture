# Results

TensorBoard logs for the behavior-cloning and PPO runs behind the final
competition submissions.

## Final submissions

| Submission | Checkpoint | Public score |
| --- | --- | --- |
| `ppo-overnight-1260` | `ppo-overnight`, iteration 1260 | 2853.4 |
| `ppo-overnight-lr3-3379` | `ppo-overnight-lr3`, iteration 3379 | 2793.9 |
| `ppo-overnight-lr3-3085` | `ppo-overnight-lr3`, iteration 3085 | 2777.6 |

Kaggle counts each team's two most recent submissions. Those were the two
`ppo-overnight-lr3` checkpoints, which placed 12th of 10,246 teams on the public
leaderboard (snapshot of 2026-10-02 UTC).

## Lineage

```text
bc ──► ppo-overnight (iterations 1–2535) ──► ppo-overnight-lr3 (iterations 2536–3379)
```

- **`bc`**: the production `lejepa` actor behavior-cloned for 16 epochs on
  replays of 2026-09-27 and 2026-09-28 leaderboard games between agents rated at
  least 2600 (observation schema 8, action interface 2). One step is one epoch.
- **`ppo-overnight`**: PPO from the `bc` actor with a fresh win/draw/loss
  critic. Each wave is 232 games: 168 mirror self-play games plus 64 games
  against a hardness-ranked league of the learner's own past snapshots. No
  reference agents were used. Terminal win/draw/loss reward, actor learning
  rate 5e-5, no entropy bonus.
- **`ppo-overnight-lr3`**: resumed from `ppo-overnight` iteration 2535 with
  three times the actor learning rate (1.5e-4) and a 1e-4 entropy bonus. The
  last 125 iterations anneal the actor learning rate linearly toward zero. The log here
  starts at iteration 2536; the earlier history is the `ppo-overnight` log.

## Viewing

The event files (about 100 MB) are stored with [Git LFS](https://git-lfs.com/).
They are not downloaded on clone. To fetch them and start TensorBoard:

```bash
git lfs pull --include 'results/tensorboard/**' --exclude ''
uv run --extra train tensorboard --logdir results/tensorboard
```

Plot the two PPO runs together to see one continuous curve from iteration 1 to
3379.
