"""Interface-3 CPU inference emits replayable set factors and engine orders."""

from __future__ import annotations

import numpy as np
from kaggle_environments import make

from kaggriculture.entity import EntityActor, EntityConfig
from kaggriculture.policy import act_batch


def test_market_set_policy_samples_effective_values_and_valid_engine_actions() -> None:
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 73})
    states = environment.reset(2)
    observations = [state.observation for state in states]
    actor = EntityActor(
        EntityConfig(
            action_interface=3,
            model_dim=32,
            attention_heads=4,
            attention_kv_heads=2,
            farm_blocks=1,
            core_layers=1,
        )
    )

    step = act_batch(
        actor,
        observations,
        [observations[1]["private"], observations[0]["private"]],
        deterministic=True,
    )

    factors = step.factors
    assert factors.market_set_values is not None
    assert factors.market_set_masks is not None
    assert factors.market_set_active is not None
    assert factors.market_set_logprobs is not None
    assert factors.market_set_values.shape == (2, 21)
    assert factors.market_set_masks.shape == (2, 21, 101)
    assert factors.market_set_active.shape == factors.market_set_values.shape
    np.testing.assert_array_equal(
        factors.market_set_masks[np.arange(2)[:, None], np.arange(21), factors.market_set_values],
        True,
    )
    assert all(len(action["market"]) <= 10 for action in step.actions)
    result = environment.step(step.actions)
    assert all(state.status != "ERROR" for state in result)
