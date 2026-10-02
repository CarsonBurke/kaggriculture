#!/usr/bin/env python
"""Compile the public v27 agent's hardcoded action script into a Rust table.

The `public-v27` agent is not a policy. It replays one precomputed action per
step from a compressed blob, keyed by `observation.step`, and only three thin
layers react to state at all: a weed repair that substitutes DIG and delays the
affected unit, a reordering of its own SELL slots by price impact, and padding
to the live unit count. Everything else about it is fixed before the episode
starts.

That makes it the one opponent worth having inside the batched wave: it is the
strength the leaderboard actually fields, it is exactly reproducible, and it
costs a table lookup instead of a Python call. This script performs the half of
that port which must not be done by hand -- turning 719 string commands per
seat into the engine's compact codes -- and fails loudly rather than guessing
if any command falls outside the action space.

`--check` re-derives the table and compares it to the committed file, so the
generated source cannot drift from the agent it claims to mirror.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kaggriculture.actions import (
    _BUY_ANIMAL,
    _BUY_PRODUCT,
    _BUY_SEED,
    _MOVE_COMMAND,
    _PICKUP_SPEC,
    _PLACE_ANIMAL,
    _PLANT_CROP,
    _SELL_PRODUCT,
    MarketKind,
    UnitAction,
)
from kaggriculture.constants import MAX_MARKET_ORDERS, MAX_UNITS, QUANTITY_BINS
from kaggriculture.opponents import PUBLIC_V27_OPPONENT

#: Operations the engine dispatches on `action[0]` alone. v27 writes a
#: redundant argument on two of them -- `["FEED", "WHEAT"]` and
#: `["FERTILIZE", "FERTILIZER"]` -- which the engine never reads: its FEED
#: branch spends a hardcoded WHEAT and its FERTILIZE branch a hardcoded
#: FERTILIZER. Dropping the argument is exact, not a normalization.
_BARE_OPERATIONS = frozenset(
    (
        "WATER",
        "HARVEST",
        "FERTILIZE",
        "DIG",
        "BUILD_COOP",
        "BUILD_PASTURE",
        "FEED",
        "COLLECT_FERTILIZER",
        "CARE",
    )
)

_GENERATED_RELATIVE = Path("rust/kagg_env/src/v27_script.rs")


def _load_agent_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("kaggriculture_public_v27", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import the public v27 agent from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _unit_action_codes() -> dict[tuple[Any, ...], int]:
    """Invert the engine's unit-action decoding into command -> code."""
    codes: dict[tuple[Any, ...], int] = {("PASS",): int(UnitAction.PASS)}
    for action, command in _MOVE_COMMAND.items():
        codes[(command,)] = int(action)
    codes[("DROP",)] = int(UnitAction.DROP)
    for action, (item, quantity) in _PICKUP_SPEC.items():
        codes[("PICKUP", item, quantity)] = int(action)
    for action, animal in _PLACE_ANIMAL.items():
        codes[("PLACE", animal)] = int(action)
    for action, crop in _PLANT_CROP.items():
        codes[("PLANT", crop)] = int(action)
    for name in _BARE_OPERATIONS:
        codes[(name,)] = int(UnitAction[name])
    return codes


def _market_kind_codes() -> dict[tuple[Any, ...], int]:
    """Invert the engine's market decoding into (operation, item) -> kind."""
    codes: dict[tuple[Any, ...], int] = {
        ("HIRE",): int(MarketKind.HIRE),
        ("BUY_LAND",): int(MarketKind.BUY_LAND),
    }
    for kind, crop in _BUY_SEED.items():
        codes[("BUY_SEED", crop)] = int(kind)
    for kind, item in _BUY_PRODUCT.items():
        codes[("BUY_PRODUCT", item)] = int(kind)
    for kind, animal in _BUY_ANIMAL.items():
        codes[("BUY_ANIMAL", animal)] = int(kind)
    for kind, product in _SELL_PRODUCT.items():
        codes[("SELL", product)] = int(kind)
    return codes


def _encode_unit(command: Any, codes: dict[tuple[Any, ...], int], where: str) -> int:
    if not isinstance(command, (list, tuple)) or not command:
        raise ValueError(f"{where}: unit command is empty")
    parts = tuple(command)
    operation = str(parts[0])
    if operation in _BARE_OPERATIONS:
        # The engine reads only the operation for these, so a trailing argument
        # is inert. Refusing it instead would reject the real agent's bytes.
        parts = (operation,)
    if parts not in codes:
        raise ValueError(f"{where}: no engine action encodes {list(command)!r}")
    return codes[parts]


def _encode_market(order: Any, codes: dict[tuple[Any, ...], int], where: str) -> tuple[int, int]:
    if not isinstance(order, (list, tuple)) or not order:
        raise ValueError(f"{where}: market order is empty")
    operation = str(order[0])
    if operation in {"HIRE", "BUY_LAND"}:
        if len(order) != 1:
            raise ValueError(f"{where}: {operation} takes no argument, got {list(order)!r}")
        return codes[(operation,)], 0
    if len(order) != 3:
        raise ValueError(f"{where}: expected [operation, item, quantity], got {list(order)!r}")
    key = (operation, str(order[1]))
    if key not in codes:
        raise ValueError(f"{where}: no engine order encodes {list(order)!r}")
    quantity = int(order[2])
    if quantity not in _QUANTITY_INDEX:
        raise ValueError(
            f"{where}: quantity {quantity} is outside the engine's bins "
            f"{QUANTITY_BINS[0]}..{QUANTITY_BINS[-1]}"
        )
    return codes[key], _QUANTITY_INDEX[quantity]


_QUANTITY_INDEX = {int(value): index for index, value in enumerate(QUANTITY_BINS)}


def compile_script(path: Path) -> tuple[list[dict[str, list[int]]], str]:
    """Return one compact row per scripted step, plus the agent file's digest."""
    module = _load_agent_module(path)
    legacy = module._LEGACY_ACTIONS
    # Both regimes alias one table in the shipped agent. If a future copy splits
    # them, a single table would silently mirror only one of the two.
    if module._REBALANCE_ACTIONS is not legacy:
        raise ValueError("this v27 copy carries two distinct action tables; the port assumes one")
    unit_codes = _unit_action_codes()
    market_codes = _market_kind_codes()
    rows: list[dict[str, list[int]]] = []
    for step, entry in enumerate(legacy):
        where = f"step {step}"
        farmer = entry.get("farmer") or ["PASS"]
        hands = list(entry.get("hands") or [])
        orders = list(entry.get("market") or [])
        if len(hands) + 1 > MAX_UNITS:
            raise ValueError(f"{where}: {len(hands)} hands exceeds {MAX_UNITS - 1} slots")
        if len(orders) > MAX_MARKET_ORDERS:
            raise ValueError(f"{where}: {len(orders)} orders exceeds {MAX_MARKET_ORDERS} slots")
        units = [int(UnitAction.PASS)] * MAX_UNITS
        units[0] = _encode_unit(farmer, unit_codes, f"{where} farmer")
        for index, command in enumerate(hands):
            units[index + 1] = _encode_unit(command, unit_codes, f"{where} hand {index}")
        kinds = [int(MarketKind.STOP)] * MAX_MARKET_ORDERS
        quantities = [0] * MAX_MARKET_ORDERS
        for index, order in enumerate(orders):
            kind, quantity = _encode_market(order, market_codes, f"{where} order {index}")
            kinds[index] = kind
            quantities[index] = quantity
        rows.append({"units": units, "market_kinds": kinds, "market_quantities": quantities})
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return rows, digest


def render_rust(rows: list[dict[str, list[int]]], digest: str, source: Path) -> str:
    def array(values: list[int]) -> str:
        return "[" + ", ".join(str(value) for value in values) + "]"

    lines = [
        "//! Generated by scripts/extract_v27_script.py -- do not edit by hand.",
        "//!",
        "//! The public `public-v27` agent replays one precomputed action per step,",
        "//! keyed by the step index, so its whole plan fits in a table. Only its weed",
        "//! repair, its SELL-slot reordering and its padding to the live unit count",
        "//! read the game at all; those live in `core.rs` beside the other built-ins.",
        "//!",
        "//! Regenerate with `--check` in the test gate so this cannot drift from the",
        "//! agent file it mirrors.",
        "//!",
        "//! The table is derived from a public Kaggle notebook by kaitofukami, used",
        "//! under the Apache License 2.0; see THIRD_PARTY_NOTICES.md.",
        "",
        "use crate::core::{MAX_MARKET_ORDERS, MAX_UNITS};",
        "",
        "/// sha256 of the agent file this table was compiled from.",
        "pub const V27_SOURCE_SHA256: &str =",
        f'    "{digest}";',
        "",
        "/// Basename of that file, recorded so a mismatch names what to look at.",
        f'pub const V27_SOURCE_NAME: &str = "{source.name}";',
        "",
        f"pub const V27_STEPS: usize = {len(rows)};",
        "",
        "#[derive(Clone, Copy, Debug)]",
        "pub struct V27Step {",
        "    /// Index 0 is the farmer and index `i` is hand `i - 1`. Slots past the",
        "    /// script's own hand list hold PASS, which is exactly what the agent's",
        "    /// `_align_hands` pads short rows with.",
        "    pub units: [u8; MAX_UNITS],",
        "    pub market_kinds: [u8; MAX_MARKET_ORDERS],",
        "    /// Exact quantity index: 0..99 decodes to 1..100.",
        "    pub market_quantities: [u8; MAX_MARKET_ORDERS],",
        "}",
        "",
        "// One line per step keeps 719 plan entries readable as a table. Formatting",
        "// each into a five-line struct would inflate the file without adding a fact.",
        "#[rustfmt::skip]",
        "pub static V27_SCRIPT: [V27Step; V27_STEPS] = [",
    ]
    for row in rows:
        lines.append(
            "    V27Step { units: "
            + array(row["units"])
            + ", market_kinds: "
            + array(row["market_kinds"])
            + ", market_quantities: "
            + array(row["market_quantities"])
            + " },"
        )
    lines.append("];")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=PUBLIC_V27_OPPONENT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the committed table differs from a fresh extraction",
    )
    args = parser.parse_args()
    agent = args.agent.expanduser().resolve()
    if not agent.is_file():
        raise SystemExit(f"public v27 agent is unavailable: {agent}")
    root = Path(__file__).resolve().parent.parent
    output = (args.output or (root / _GENERATED_RELATIVE)).resolve()
    rows, digest = compile_script(agent)
    rendered = render_rust(rows, digest, agent)
    if args.check:
        if not output.is_file():
            raise SystemExit(f"generated table is missing: {output}")
        current = output.read_text(encoding="utf-8")
        if current != rendered:
            raise SystemExit(
                f"{output} is stale; regenerate with scripts/extract_v27_script.py "
                f"(agent sha256 {digest})"
            )
        print(f"table matches {agent.name} ({len(rows)} steps, sha256 {digest[:12]})")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"wrote {output} ({len(rows)} steps) from {agent.name} sha256 {digest[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
