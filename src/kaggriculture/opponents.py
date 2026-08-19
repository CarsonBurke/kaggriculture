"""External opponent registry shared by every evaluation entry point."""

from __future__ import annotations

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
PUBLIC_V27_OPPONENT = Path("/var/tmp/kaggriculture-kaito-v27-main.py")
PUBLIC_V27_ALIASES = frozenset(("v27", "public-v27"))
PUBLIC_V16_TEACHER = Path("/var/tmp/kaggriculture-boatlee-v16-rc5-main.py")
PUBLIC_V16_ALIASES = frozenset(("v16", "public-v16"))


def normalize_opponent(opponent: str) -> tuple[str, str]:
    """Resolve an opponent spec to a stable label and a runnable reference.

    Built-in engine agents pass through by name; the public v27 and v16 aliases
    pin the known local copies; anything else must be an existing Python agent file.

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
    path = Path(opponent).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"opponent does not exist: {path}")
    return path.name, str(path)
