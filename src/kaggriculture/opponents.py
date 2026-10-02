"""External opponent registry shared by every evaluation entry point."""

from __future__ import annotations

import os
from pathlib import Path

# Ordered because the batched Rust wave addresses these agents by code
# (`BuiltinAgent::from_code`), and the code is this tuple's index plus one.
#
# `scripted-v27` is the in-engine port of the public v27 agent, whose whole plan
# is one hardcoded action per step. It is deliberately *not* named `v27`: the
# aliases below must keep resolving to the reference Python file, so an
# evaluation measures the real opponent and never the port that mirrors it.
BUILTIN_AGENT_ORDER = ("pass", "random", "starter", "scripted-v27")
BUILTIN_OPPONENTS = frozenset(BUILTIN_AGENT_ORDER)


def _agent_dir() -> Path:
    """Where this machine keeps its copies of other competitors' public agents.

    The files are not ours to redistribute, so they live outside the checkout:
    a frozen source snapshot must resolve the same copies as the live tree.
    """
    configured = os.environ.get("KAGGRICULTURE_AGENT_DIR")
    if configured:
        # Resolved now: a runnable reference must be absolute, and workers run
        # from other directories.
        return Path(configured).expanduser().resolve()
    # The XDG spec says a relative XDG_DATA_HOME is invalid and must be ignored.
    data_home = Path(os.environ.get("XDG_DATA_HOME", ""))
    if not data_home.is_absolute():
        data_home = Path.home() / ".local" / "share"
    return data_home / "kaggriculture" / "agents"


AGENT_DIR = _agent_dir()
PUBLIC_V27_OPPONENT = AGENT_DIR / "kaggriculture-kaito-v27-main.py"
PUBLIC_V27_ALIASES = frozenset(("v27", "public-v27"))
PUBLIC_V16_TEACHER = AGENT_DIR / "kaggriculture-boatlee-v16-rc5-main.py"
PUBLIC_V16_ALIASES = frozenset(("v16", "public-v16"))

# Dynamic reference agents: public Kaggle agents whose play reacts to the state,
# pinned as machine-local copies like the references above. An official-engine
# round robin ranked them (artifacts/probes/teachers-20260930/round-robin.jsonl:
# fourteen agents, four seeds, both seats); agents that played identically there
# are one family and appear once, under its first member (flex and cha22 are
# demand-timing, top-2-master-v4 is demand-preserving). Every one matches the
# official engine exactly when played natively. Families of near-identical play
# stay on one side of the split below.
REFERENCE_AGENT_DIR = AGENT_DIR / "reference"
REFERENCE_AGENTS = (
    "demand-preserving",
    "demand-timing",
    "demand-advance4",
    "hybrid-2965",
    "harvest-ledger",
    "idle-seller",
    "shepherds-ledger",
    "master-engine-v53",
    "bronze-v31",
    "kaito-v48",
)
# What the learner trains against as fixed league lanes...
LEAGUE_REFERENCE_AGENTS = (
    "demand-timing",
    "hybrid-2965",
    "harvest-ledger",
    "master-engine-v53",
    "bronze-v31",
)
# ...and what it is scored against and never trains on, so a gain there is
# transfer. It holds the strongest family (demand-preserving, first in the
# round robin), demand-advance4, the idle-seller/shepherds family, and kaito-v48,
# whose early sales collapse the prices a neural agent sells into.
HELDOUT_REFERENCE_AGENTS = (
    "demand-preserving",
    "demand-advance4",
    "idle-seller",
    "shepherds-ledger",
    "kaito-v48",
)


def reference_agent_path(name: str) -> Path:
    """The pinned copy of a named reference agent."""
    if name not in REFERENCE_AGENTS:
        raise KeyError(f"not a reference agent: {name!r}")
    path = REFERENCE_AGENT_DIR / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"reference agent {name} is unavailable: {path}")
    return path


def normalize_opponent(opponent: str) -> tuple[str, str]:
    """Resolve an opponent spec to a stable label and a runnable reference.

    Built-in engine agents pass through by name; the public v27 and v16 aliases
    and the reference agents' names pin the known local copies; anything else
    must be an existing Python agent file.

    Labels are display names and may collide with built-ins (an agent file
    literally named ``starter``). Consumers deciding whether a digest exists
    must test the *runnable* against ``BUILTIN_OPPONENTS`` — a file opponent's
    runnable is always an absolute path and never a built-in name.
    """
    if opponent in BUILTIN_OPPONENTS:
        return opponent, opponent
    if opponent in PUBLIC_V27_ALIASES:
        path = PUBLIC_V27_OPPONENT.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"public v27 opponent is unavailable: {path}")
        return "public-v27", str(path)
    if opponent in PUBLIC_V16_ALIASES:
        path = PUBLIC_V16_TEACHER.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"public v16 teacher is unavailable: {path}")
        return "public-v16", str(path)
    if opponent in REFERENCE_AGENTS:
        return opponent, str(reference_agent_path(opponent))
    path = Path(opponent).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"opponent does not exist: {path}")
    return path.name, str(path)
