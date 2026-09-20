"""Deterministic checkpoint inference used by local evaluation and Kaggle bundles."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kaggriculture.orientation import Orientation
from kaggriculture.policy import act_batch
from kaggriculture.provenance import (
    is_legacy_run_provenance,
    validate_run_provenance,
    validate_source_identity,
)
from kaggriculture.registry import resolve_architecture

ACTOR_ARTIFACT_FORMAT_VERSION = 5
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
CHECKPOINT_FORMAT_VERSION = 12
# Versions before 12 carry no orientation code. On the actor-only read path
# their payloads are otherwise unchanged, so the legacy versions stay readable
# with orientation defaulting to IDENTITY -- which is what those runs played
# under -- while resume above refuses them.
LEGACY_CHECKPOINT_FORMAT_VERSIONS = frozenset((7, 8, 9, 10, 11))
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
        # Evaluation is the real board. A stored member code is history.
        "orientation": int(Orientation.IDENTITY),
    }


def load_actor_artifact(
    path: Path, device: torch.device | str = "cpu", *, agent: int | None = None
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

        # Training may have cycled frames; the competition board is identity.
        # A stored member code on a legacy artifact is ignored.
        self.orientation = Orientation.IDENTITY

    def act_many(self, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Act on independent environments in one model forward."""
        actions = act_batch(
            self.actor,
            observations,
            deterministic=True,
            orientation=self.orientation,
        ).actions
        return [
            clear_standing_weeds(observation, action)
            for observation, action in zip(observations, actions, strict=True)
        ]

    def __call__(self, observation: dict[str, Any]) -> dict[str, Any]:
        return self.act_many([observation])[0]


def _tile_at(farm: dict[str, Any], position: Any) -> Any:
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return None
    tiles = farm.get("tiles") or []
    y = int(position[1])
    x = int(position[0])
    if y < 0 or y >= len(tiles) or x < 0 or x >= len(tiles[y]):
        return None
    return tiles[y][x]


def _is_weed(tile: Any) -> bool:
    return isinstance(tile, dict) and tile.get("kind") == "WEED"


def clear_standing_weeds(observation: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    """Force DIG for every live unit standing on a weed.

    Mutates and returns ``action``. Units the engine will not execute this
    turn are left alone: a leftover hand command on a missing unit is not
    a standing tile.
    """
    player = int(observation.get("player", 0) or 0)
    farms = observation.get("farms") or []
    if player >= len(farms) or not isinstance(action, dict):
        return action
    farm = farms[player]
    if _is_weed(_tile_at(farm, farm.get("farmer"))):
        action["farmer"] = ["DIG"]
    hands = list(action.get("hands") or [])
    positions = list(farm.get("hands") or [])
    for index, position in enumerate(positions):
        if index >= len(hands):
            break
        if _is_weed(_tile_at(farm, position)):
            hands[index] = ["DIG"]
    action["hands"] = hands
    return action
