"""Deterministic checkpoint inference used by local evaluation and Kaggle bundles."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any

import torch

from kaggriculture.model import FarmActor, ModelConfig
from kaggriculture.policy import act_batch
from kaggriculture.provenance import validate_run_provenance, validate_source_identity

ACTOR_ARTIFACT_FORMAT_VERSION = 4
CHECKPOINT_FORMAT_VERSION = 4


def actor_artifact_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    checkpoint_version = checkpoint.get("format_version")
    if checkpoint_version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format: {checkpoint_version}; "
            f"expected {CHECKPOINT_FORMAT_VERSION}"
        )
    if "actor" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError("checkpoint is missing actor weights or model configuration")
    identity = validate_source_identity(checkpoint.get("source_identity"))
    run_provenance = validate_run_provenance(checkpoint.get("run_provenance"))
    if run_provenance is not None and run_provenance["source_identity"] != identity:
        raise ValueError("checkpoint run provenance source does not match source identity")
    return {
        "format_version": ACTOR_ARTIFACT_FORMAT_VERSION,
        "model_config": checkpoint["model_config"],
        "actor": checkpoint["actor"],
        "iteration": int(checkpoint.get("iteration", 0)),
        "metrics": checkpoint.get("metrics", {}),
        "source_identity": identity,
        "run_provenance": run_provenance,
    }


def load_actor_artifact(
    path: Path, device: torch.device | str = "cpu"
) -> tuple[FarmActor, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    version = payload.get("format_version")
    if version != ACTOR_ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported actor artifact format: {version}; "
            f"expected {ACTOR_ARTIFACT_FORMAT_VERSION}"
        )
    identity = validate_source_identity(payload.get("source_identity"))
    run_provenance = validate_run_provenance(payload.get("run_provenance"))
    if run_provenance is not None and run_provenance["source_identity"] != identity:
        raise ValueError("actor artifact run provenance source does not match source identity")
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
