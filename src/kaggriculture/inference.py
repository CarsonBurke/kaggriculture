"""Deterministic checkpoint inference used by local evaluation and Kaggle bundles."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kaggriculture.policy import act_batch
from kaggriculture.provenance import (
    is_legacy_run_provenance,
    validate_run_provenance,
    validate_source_identity,
)
from kaggriculture.registry import resolve_architecture

ACTOR_ARTIFACT_FORMAT_VERSION = 5
# Version 10 renamed the resume payload's `vapo_config` key to `ppo_config`.
# The bump is the whole point of the number: the key has exactly one reader and
# it subscripts, so without it a version-9 checkpoint passes the format check,
# gets its weights, optimizers and every RNG stream restored, and only then
# dies on a bare KeyError with the process already mutated. Version 9 added
# required PFSP league score-rate resume state, the critic_epochs knob, and the
# architecture tag. Resume (training.py) demands the current version exactly;
# actor export stays readable across the legacy versions for weights and model
# configuration, which is what the rename left untouched.
#
# Their calibration provenance is a different matter and is NOT carried across.
# It is versioned separately and independently of the checkpoint format, and
# every superseded version is refused rather than migrated, because each bump
# so far removed a claim the older record could not substantiate: version 1
# held a single `compile_models` whose per-phase meaning the run could not have
# measured, and version 2 held two per-phase speedups differenced across a pair
# of runs that moved both knobs at once, attributing to each knob a change the
# evidence cannot separate. Exporting either would let an artifact assert a
# calibration nobody can recompute, which is the exact failure the version bump
# exists to prevent -- so such a checkpoint is refused at the export boundary
# rather than being migrated or silently stripped.
CHECKPOINT_FORMAT_VERSION = 10
LEGACY_CHECKPOINT_FORMAT_VERSIONS = frozenset((7, 8, 9))
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
    try:
        run_provenance = validate_run_provenance(checkpoint.get("run_provenance"))
    except ValueError as error:
        # Name the incompatibility at the boundary the caller is standing on.
        # Without this the operator exporting a superseded checkpoint sees a
        # bare "unsupported run provenance format: N" from a nested validator
        # and reads it as corruption rather than as an artifact whose recorded
        # calibration evidence no longer substantiates its own decision.
        raise ValueError(
            f"checkpoint format {checkpoint_version} carries superseded calibration "
            f"provenance that cannot be exported: {error}"
        ) from error
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
    # Loading weights and carrying a calibration claim forward are different
    # operations, and only the second one needs the claim to be interpretable.
    # This function is the read path for cross-tree work that deliberately does
    # not require identity equality -- `--init-actor-from`, replay viewing,
    # behavior audits -- so refusing an artifact because its *compile
    # calibration* predates the rollout/update split would reject perfectly
    # good weights over a field none of those callers read. The run being
    # started records its own calibration; the source artifact's is history.
    #
    # `actor_artifact_from_checkpoint` stays strict, because that is the path
    # that copies provenance into a submission, where an uninterpretable claim
    # would be asserted as though it were recoverable.
    # Narrowly: a record that merely predates the current format is dropped;
    # anything else is still validated and still raises. Swallowing every
    # ValueError here would discard tamper-evidence on the read path, which is
    # a much worse trade than the one being made.
    stored = payload.get("run_provenance")
    if is_legacy_run_provenance(stored):
        stored = None
    run_provenance = validate_run_provenance(stored)
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
