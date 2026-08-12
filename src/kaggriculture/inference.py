"""Deterministic checkpoint inference used by local evaluation and Kaggle bundles."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any

import torch

from kaggriculture.model import FarmActor, ModelConfig
from kaggriculture.policy import act_batch


def actor_artifact_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    if "actor" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError("checkpoint is missing actor weights or model configuration")
    return {
        "format_version": 2,
        "model_config": checkpoint["model_config"],
        "actor": checkpoint["actor"],
        "iteration": int(checkpoint.get("iteration", 0)),
        "metrics": checkpoint.get("metrics", {}),
    }


def load_actor_artifact(
    path: Path, device: torch.device | str = "cpu"
) -> tuple[FarmActor, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format_version") not in {1, 2}:
        raise ValueError(f"unsupported actor artifact format: {payload.get('format_version')}")
    config = ModelConfig(**payload["model_config"])
    actor = FarmActor(config).to(device)
    actor.load_state_dict(payload["actor"])
    actor.eval()
    return actor, payload


class CheckpointAgent:
    """Callable deterministic agent with no cross-episode mutable policy state."""

    def __init__(
        self,
        artifact: Path,
        *,
        device: torch.device | str = "cpu",
        torch_threads: int = 1,
    ) -> None:
        if torch_threads > 0:
            torch.set_num_threads(torch_threads)
            with suppress(RuntimeError):
                # PyTorch only permits changing this before the first parallel op.
                torch.set_num_interop_threads(1)
        self.actor, self.metadata = load_actor_artifact(artifact, device)

    def __call__(self, observation: dict[str, Any]) -> dict[str, Any]:
        return act_batch(
            self.actor,
            None,
            [observation],
            deterministic=True,
        ).actions[0]
