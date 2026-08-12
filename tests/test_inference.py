from __future__ import annotations

from pathlib import Path

import torch

from kaggriculture.inference import (
    actor_artifact_from_checkpoint,
    load_actor_artifact,
)
from kaggriculture.model import FarmActor, ModelConfig


def test_actor_artifact_round_trip(tmp_path: Path) -> None:
    config = ModelConfig(width=16, residual_blocks=1, hidden=32, query_features=8)
    actor = FarmActor(config)
    artifact = actor_artifact_from_checkpoint(
        {
            "model_config": config.to_dict(),
            "actor": actor.state_dict(),
            "iteration": 3,
        }
    )
    path = tmp_path / "model.pt"
    torch.save(artifact, path)

    restored, metadata = load_actor_artifact(path)

    assert metadata["iteration"] == 3
    for expected, actual in zip(actor.parameters(), restored.parameters(), strict=True):
        assert torch.equal(expected, actual)
