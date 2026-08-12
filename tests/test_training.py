from __future__ import annotations

import numpy as np
import torch

from kaggriculture.model import DistributionalCritic, FarmActor, ModelConfig
from kaggriculture.training import load_checkpoint, save_checkpoint
from kaggriculture.vapo import VapoConfig, make_optimizers


def test_checkpoint_round_trips_local_training_generator(tmp_path) -> None:
    model_config = ModelConfig(width=8, residual_blocks=1, hidden=16, query_features=4)
    vapo_config = VapoConfig(epochs=1, minibatch_size=4, use_bfloat16=False)
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    actor_optimizer, critic_optimizer = make_optimizers(actor, critic, vapo_config)
    generator = np.random.default_rng(17)
    generator.random(5)
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path,
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        model_config=model_config,
        vapo_config=vapo_config,
        iteration=3,
        next_seed=41,
        metrics={"score_rate": 0.75},
        training_rng_state=generator.bit_generator.state,
    )
    expected = generator.random(8)

    payload = load_checkpoint(
        path,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        device=torch.device("cpu"),
    )
    restored = np.random.default_rng()
    restored.bit_generator.state = payload["training_rng"]

    assert payload["iteration"] == 3
    assert payload["next_seed"] == 41
    assert restored.random(8).tolist() == expected.tolist()
