# kraggiculture

Research and evaluation tooling for the Kaggriculture simulation competition.

The learner is direct, from-scratch VAPO self-play. Training does not depend on
expert demonstrations, distillation, behavior cloning, or value pretraining. An
exact batched Rust simulator supplies high-throughput rollouts; the pinned Kaggle
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
