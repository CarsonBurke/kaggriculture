"""Deterministic checkpoint inference used by local evaluation and Kaggle bundles."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from kaggriculture.orientation import Orientation
from kaggriculture.policy import act_batch, prepare_quantity_heads
from kaggriculture.provenance import (
    is_legacy_run_provenance,
    validate_run_provenance,
    validate_source_identity,
)
from kaggriculture.registry import resolve_architecture

ACTOR_ARTIFACT_FORMAT_VERSION = 5
# Version 13 adds a critic-side structured dynamics predictor and its optimizer
# to full recovery checkpoints. Both dynamics modules remain training-only, so
# actor export can still read version 12 as well as the new format without
# carrying either predictor forward.
#
# Version 12 records each member's `orientation`: the grid symmetry its
# observations were rendered through during training. Acting under anything
# else shows the weights a world they never saw, so evaluation and submission
# replay the recorded code, and resume demands this version rather than
# guessing a rendering the run never used.
#
# Version 11 gave a population run's payload a list of members under `agents`
# in place of the four top-level state dicts a single learner keeps. The bump is
# what stops a version-10 resume from being read as a population of one: the
# resume path reconstructs N actors, critics and optimizer pairs from that list,
# and a payload without it would restore nothing for members 1..N-1 and train
# them from their fresh initialization while reporting a resumed run.
#
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
CHECKPOINT_FORMAT_VERSION = 13
# Versions before 13 remain readable on the actor-only path because the recovery
# additions do not change actor weights or model configuration. Resume demands
# the current version exactly and never guesses absent training state.
LEGACY_CHECKPOINT_FORMAT_VERSIONS = frozenset((7, 8, 9, 10, 11, 12))
SUPPORTED_CHECKPOINT_FORMAT_VERSIONS = LEGACY_CHECKPOINT_FORMAT_VERSIONS | {
    ACTOR_ARTIFACT_FORMAT_VERSION,
    CHECKPOINT_FORMAT_VERSION,
}
SUPPORTED_ACTOR_INPUT_FORMAT_VERSIONS = SUPPORTED_CHECKPOINT_FORMAT_VERSIONS

#: Where a population checkpoint keeps its members. Its presence is what tells
#: the two payload shapes apart, and the distinction is deliberately not papered
#: over: a single learner's payload keeps the four top-level state dicts it has
#: always had, and a population payload has NO top-level actor at all. Aliasing
#: member zero there would let every reader of "the actor" -- submission
#: selection, external evaluation, replay viewing, export -- quietly score one
#: arbitrary member and report it as the run's strength.
POPULATION_CHECKPOINT_KEY = "agents"


def checkpoint_agent_count(checkpoint: Mapping[str, Any]) -> int:
    """How many members a checkpoint carries; one for a single learner."""
    members = checkpoint.get(POPULATION_CHECKPOINT_KEY)
    return len(members) if isinstance(members, list) else 1


def checkpoint_actor_state(checkpoint: Mapping[str, Any], agent: int | None) -> dict[str, Any]:
    """The actor weights `agent` names, refusing a read that would be arbitrary.

    A population payload has no single answer to "the actor", so `agent is None`
    against one is an error rather than member zero. The two directions are both
    checked because either mistake is silent: reading a population as a single
    learner reports one member as the run, and naming an agent against a single
    learner is a caller that believes it selected something.
    """
    members = checkpoint.get(POPULATION_CHECKPOINT_KEY)
    if isinstance(members, list):
        available = f"0..{len(members) - 1}"
        if agent is None:
            raise ValueError(
                f"checkpoint holds a population of {len(members)} agents and has no "
                f"single actor; select one with agent={available}"
            )
        if not 0 <= agent < len(members):
            raise ValueError(
                f"checkpoint population has no agent {agent}; available agents are {available}"
            )
        return members[agent]["actor"]
    if agent is not None:
        raise ValueError(
            f"checkpoint holds a single learner, not a population, so agent {agent} "
            "does not name anything in it"
        )
    if "actor" not in checkpoint:
        raise ValueError("checkpoint is missing actor weights")
    return checkpoint["actor"]


def _orientation_code(value: Any) -> Orientation:
    try:
        return Orientation(int(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid orientation code: {value!r}") from error


def checkpoint_orientation(checkpoint: Mapping[str, Any], agent: int | None = None) -> Orientation:
    """Historical per-member code, ignored at play time.

    Older population checkpoints stamped member *i*'s cycle index beside its
    weights. Training now cycles frames per game and evaluation always plays
    the real board, so this reader exists only to keep old files loadable.
    Callers that act must use ``Orientation.IDENTITY``.
    """
    members = checkpoint.get(POPULATION_CHECKPOINT_KEY)
    if isinstance(members, list):
        if agent is None or not 0 <= agent < len(members):
            available = f"0..{len(members) - 1}" if members else "none"
            raise ValueError(
                f"checkpoint population holds no member {agent}; select one with agent={available}"
            )
        return _orientation_code(members[agent].get("orientation", 0))
    if agent is not None:
        raise ValueError(
            f"checkpoint holds a single learner, not a population, so agent {agent} "
            "does not name anything in it"
        )
    return _orientation_code(checkpoint.get("orientation", 0))


def actor_artifact_from_checkpoint(
    checkpoint: dict[str, Any], *, agent: int | None = None
) -> dict[str, Any]:
    checkpoint_version = checkpoint.get("format_version")
    if checkpoint_version not in SUPPORTED_CHECKPOINT_FORMAT_VERSIONS:
        expected = ", ".join(map(str, sorted(SUPPORTED_CHECKPOINT_FORMAT_VERSIONS)))
        raise ValueError(
            f"unsupported checkpoint format: {checkpoint_version}; expected one of {expected}"
        )
    if "model_config" not in checkpoint:
        raise ValueError("checkpoint is missing actor weights or model configuration")
    actor_state = checkpoint_actor_state(checkpoint, agent)
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
        "actor": actor_state,
        "iteration": int(checkpoint.get("iteration", 0)),
        "metrics": checkpoint.get("metrics", {}),
        "source_identity": identity,
        "run_provenance": run_provenance,
        **{
            key: checkpoint[key]
            for key in ("seed_usage", "bc_provenance", "initial_actor")
            if key in checkpoint
        },
        # Evaluation is the real board. A stored member code is history.
        "orientation": int(Orientation.IDENTITY),
    }


def _cpu_portable_structured_state(
    model_config: Mapping[str, Any],
    state: Mapping[str, Tensor],
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Convert bias-free fused MLP parameters into ordinary CPU Linear layers.

    The layouts are algebraically equivalent, not numerically identical across
    CUDA BF16 and CPU FP32. Callers use this path specifically to measure the
    policy that can execute in Kaggle rather than treating CUDA evaluation as a
    proxy for it.
    """
    if not model_config.get("fused_mlp", False):
        raise ValueError("actor does not use fused structured MLPs")
    fused_payload = {
        "architecture": "structured",
        "model_config": dict(model_config),
    }
    fused_actor = resolve_architecture(fused_payload).build_actor(dict(model_config))
    fused_actor.load_state_dict(state, strict=True)

    converted: dict[str, Tensor] = {}
    up_prefixes: set[str] = set()
    down_prefixes: set[str] = set()

    def store(name: str, value: Tensor) -> None:
        if name in converted:
            raise ValueError(f"fused MLP conversion would overwrite state key {name!r}")
        converted[name] = value.detach().to(device="cpu", copy=True)

    for name, value in state.items():
        if name.endswith(".ffn.up_weight"):
            prefix = name.removesuffix("up_weight")
            store(f"{prefix}input.weight", value)
            store(f"{prefix}input.bias", value.new_zeros(value.shape[0]))
            up_prefixes.add(prefix)
        elif name.endswith(".ffn.down_weight"):
            prefix = name.removesuffix("down_weight")
            store(f"{prefix}output.weight", value.detach().T.contiguous())
            store(f"{prefix}output.bias", value.new_zeros(value.shape[1]))
            down_prefixes.add(prefix)
        else:
            store(name, value)
    if not up_prefixes:
        raise ValueError("fused actor contains no fused MLP weights")
    if up_prefixes != down_prefixes:
        missing_up = sorted(down_prefixes - up_prefixes)
        missing_down = sorted(up_prefixes - down_prefixes)
        raise ValueError(
            f"incomplete fused MLP state: missing up={missing_up}, missing down={missing_down}"
        )

    portable_config = dict(model_config)
    portable_config["fused_mlp"] = False
    portable_payload = {
        "architecture": "structured",
        "model_config": portable_config,
    }
    portable_actor = resolve_architecture(portable_payload).build_actor(portable_config)
    portable_actor.load_state_dict(converted, strict=True)
    return portable_config, converted


def cpu_portable_actor(
    checkpoint: Mapping[str, Any],
    *,
    agent: int | None = None,
) -> nn.Module:
    """Build the CPU policy that a fused structured checkpoint can submit."""
    if resolve_architecture(dict(checkpoint)).name != "structured":
        raise ValueError("CPU conversion applies only to structured actors")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("actor is missing its model configuration")
    portable_config, portable_state = _cpu_portable_structured_state(
        model_config,
        checkpoint_actor_state(checkpoint, agent),
    )
    payload = {"architecture": "structured", "model_config": portable_config}
    actor = resolve_architecture(payload).build_actor(portable_config)
    actor.load_state_dict(portable_state, strict=True)
    actor.eval()
    return actor


def cpu_portable_actor_artifact(
    checkpoint: Mapping[str, Any],
    *,
    agent: int | None = None,
) -> dict[str, Any]:
    """Export the exact CPU policy shape used by submission and external evaluation."""
    artifact = actor_artifact_from_checkpoint(dict(checkpoint), agent=agent)
    portable_config, portable_state = _cpu_portable_structured_state(
        artifact["model_config"],
        artifact["actor"],
    )
    return {
        **artifact,
        "model_config": portable_config,
        "actor": portable_state,
    }


def load_actor_artifact(
    path: Path, device: torch.device | str = "cpu", *, agent: int | None = None
) -> tuple[nn.Module, dict[str, Any]]:
    target_device = torch.device(device)
    payload = torch.load(path, map_location="cpu", weights_only=False)
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

    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("actor artifact is missing its model configuration")
    fused_mlp = bool(model_config.get("fused_mlp", False))
    if fused_mlp and target_device.type != "cuda":
        raise ValueError("fused structured artifacts require CUDA inference")
    actor = resolve_architecture(payload).build_actor(dict(model_config)).to(target_device)
    actor.load_state_dict(checkpoint_actor_state(payload, agent))
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
        agent: int | None = None,
    ) -> None:
        if torch_threads > 0:
            torch.set_num_threads(torch_threads)
            with suppress(RuntimeError):
                # PyTorch only permits changing this before the first parallel op.
                torch.set_num_interop_threads(1)
        self.actor, self.metadata = load_actor_artifact(artifact, device, agent=agent)
        self.member = agent
        self.quantity_heads = prepare_quantity_heads(self.actor)
        model_config = self.metadata.get("model_config")
        self.fused_mlp = isinstance(model_config, Mapping) and bool(
            model_config.get("fused_mlp", False)
        )

        # Training may have cycled frames; the competition board is identity.
        # A stored member code on a legacy artifact is ignored.
        self.orientation = Orientation.IDENTITY

    def act_many(self, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Act on independent environments in one model forward."""
        inference_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.fused_mlp
            else nullcontext()
        )
        with inference_context:
            actions = act_batch(
                self.actor,
                observations,
                deterministic=True,
                orientation=self.orientation,
                quantity_heads=self.quantity_heads,
            ).actions
        return actions

    def __call__(self, observation: dict[str, Any]) -> dict[str, Any]:
        return self.act_many([observation])[0]
