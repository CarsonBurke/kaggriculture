from __future__ import annotations

import numpy as np
import pytest
import torch

from kaggriculture.model import FarmActor, ModelConfig
from kaggriculture.rollout import collect_mixed_play_rust


@pytest.mark.parametrize("deterministic", [False, True])
def test_builtin_lanes_preserve_other_games_sampling_streams(deterministic: bool) -> None:
    """Changing unrelated opponents cannot shift a physical game's policy draws.

    The mixed wave has one assigned neural lane, one unused frozen network and
    a wider built-in group. The reference fills that group with the unused
    network instead. Only those games may change: compacting frozen RNG rows,
    scattering a lane to the wrong seat, or borrowing the unused model for the
    assigned lane changes the retained trajectories.
    """
    config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        actor = FarmActor(config)
        opponents = (FarmActor(config), FarmActor(config))

    def collect(assignments: tuple[int, ...], builtins: tuple[str, ...]):
        return collect_mixed_play_rust(
            actor,
            opponents,
            self_play_games=1,
            league_games=4,
            opponent_indices=np.asarray(assignments, dtype=np.int64),
            builtin_lanes=builtins,
            opponent_temperatures=(0.7, 1.3),
            deterministic=deterministic,
            deterministic_opponent=deterministic,
            seed_start=77,
            sampling_seed=29,
            forward_mode="eager",
        )

    reference = collect((0, 1, 0, 0), ())
    mixed = collect((2, 1, 2, 2), ("pass",))
    # Both self-play seats and league game 1 retain exactly the same policies,
    # seeds and physical row positions. Built-ins precede and follow that game.
    retained = np.asarray([0, 1, 3])
    for name in (
        "episode_seeds",
        "valid",
        "unit_actions",
        "market_kinds",
        "market_quantities",
        "unit_masks",
        "market_kind_masks",
        "market_quantity_masks",
        "unit_active",
        "market_active",
        "market_quantity_active",
        "rewards",
        "final_money",
        "opponent_money",
        "seats",
    ):
        np.testing.assert_array_equal(
            getattr(mixed, name)[retained], getattr(reference, name)[retained]
        )
    for name in (
        "old_unit_logprobs",
        "old_market_kind_logprobs",
        "old_market_quantity_logprobs",
        "entropy_sums",
    ):
        np.testing.assert_allclose(
            getattr(mixed, name)[retained],
            getattr(reference, name)[retained],
            rtol=1e-6,
            atol=1e-6,
        )
    for name, state in reference.states.items():
        np.testing.assert_array_equal(mixed.states[name][retained], state[retained])
    np.testing.assert_array_equal(mixed.opponent_money[[2, 4, 5]], 3000.0)
