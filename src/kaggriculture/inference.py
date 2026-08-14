"""Deterministic checkpoint inference used by local evaluation and Kaggle bundles."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kaggriculture.policy import act_batch
from kaggriculture.provenance import validate_run_provenance, validate_source_identity
from kaggriculture.registry import resolve_architecture

ACTOR_ARTIFACT_FORMAT_VERSION = 5
# Version 9 added required PFSP league score-rate resume state, the
# critic_epochs knob, and the architecture tag. Resume (training.py) demands
# the current version exactly; actor export stays readable across the legacy
# versions because their actor/model_config/provenance schema is unchanged.
CHECKPOINT_FORMAT_VERSION = 9
LEGACY_CHECKPOINT_FORMAT_VERSIONS = frozenset((7, 8))
SUPPORTED_CHECKPOINT_FORMAT_VERSIONS = LEGACY_CHECKPOINT_FORMAT_VERSIONS | {
    ACTOR_ARTIFACT_FORMAT_VERSION,
    CHECKPOINT_FORMAT_VERSION,
}
SUPPORTED_ACTOR_INPUT_FORMAT_VERSIONS = SUPPORTED_CHECKPOINT_FORMAT_VERSIONS


def actor_artifact_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    checkpoint_version = checkpoint.get("format_version")
    if checkpoint_version not in SUPPORTED_CHECKPOINT_FORMAT_VERSIONS:
        expected = ", ".join(map(str, sorted(SUPPORTED_CHECKPOINT_FORMAT_VERSIONS)))
        raise ValueError(
            f"unsupported checkpoint format: {checkpoint_version}; expected one of {expected}"
        )
    if "actor" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError("checkpoint is missing actor weights or model configuration")
    identity = validate_source_identity(checkpoint.get("source_identity"))
    run_provenance = validate_run_provenance(checkpoint.get("run_provenance"))
    if run_provenance is not None and run_provenance["source_identity"] != identity:
        raise ValueError("checkpoint run provenance source does not match source identity")
    return {
        "format_version": ACTOR_ARTIFACT_FORMAT_VERSION,
        "architecture": resolve_architecture(checkpoint).name,
        "model_config": checkpoint["model_config"],
        "actor": checkpoint["actor"],
        "iteration": int(checkpoint.get("iteration", 0)),
        "metrics": checkpoint.get("metrics", {}),
        "source_identity": identity,
        "run_provenance": run_provenance,
    }


def load_actor_artifact(
    path: Path, device: torch.device | str = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    version = payload.get("format_version")
    if version not in SUPPORTED_ACTOR_INPUT_FORMAT_VERSIONS:
        expected = ", ".join(map(str, sorted(SUPPORTED_ACTOR_INPUT_FORMAT_VERSIONS)))
        raise ValueError(
            f"unsupported actor artifact format: {version}; expected one of {expected}"
        )
    identity = validate_source_identity(payload.get("source_identity"))
    run_provenance = validate_run_provenance(payload.get("run_provenance"))
    if run_provenance is not None and run_provenance["source_identity"] != identity:
        raise ValueError("actor artifact run provenance source does not match source identity")
    actor = resolve_architecture(payload).build_actor(payload["model_config"]).to(device)
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
            [observation],
            deterministic=True,
        ).actions[0]
