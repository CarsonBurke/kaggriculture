#!/usr/bin/env python3
"""Differentially gate the native scripted-v27 built-in against the real agent file.

The public v27 agent is the strength this leaderboard actually fields, and it is
now ported into the batched wave so training can face it at rollout speed. A
port of an opponent is only worth having if it is the same opponent, so this
audit refuses anything short of exact agreement.

Two independent checks run, for the same reason the built-in audit keeps two:

Per-step equality asks the real agent about the exact state the native built-in
just acted from, shaping the native snapshot into the observation it reads, and
compares the two actions as engine factor codes. It localizes a disagreement to
one step, one unit slot or one market slot, which whole-episode money cannot.

Whole-episode equality plays each pairing natively and then plays the same seed
through `kaggle_environments`' own `run`, handing it the agent file itself, and
demands the same final banks. It shares no glue with the code it audits: no
observation shaping of ours sits between the reference and its answer. This is
the check that would catch a faithful action stream applied to a state our
engine had already drifted on.

The digest gate comes first. The native table is compiled from one exact file,
and comparing the port against a copy that has since changed would prove
nothing, so the audit stops unless the bytes match what the table was built
from.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from kaggle_environments import make

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kaggriculture.actions import (
    MAX_MARKET_ORDERS,
    MarketKind,
    UnitAction,
)
from kaggriculture.constants import QUANTITY_BINS
from kaggriculture.opponents import PUBLIC_V27_OPPONENT
from kaggriculture.rust_env import load_native

PLAYERS = 2
EPISODE_STEPS = 720
#: One transition fewer than the configured episode: the last one ends the game.
MAX_TRANSITIONS = EPISODE_STEPS - 1
#: `BuiltinAgent::from_code`; `opponents.BUILTIN_AGENT_ORDER` index plus one.
BUILTIN_CODES = {"pass": 1, "random": 2, "starter": 3, "scripted-v27": 4}
#: Operations the engine dispatches on `action[0]` alone, so a trailing argument
#: is inert. v27 writes one on FEED and FERTILIZE; the engine's branches spend a
#: hardcoded WHEAT and FERTILIZER and never read it.
BARE_OPERATIONS = frozenset(
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
_QUANTITY_INDEX = {int(value): index for index, value in enumerate(QUANTITY_BINS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=PUBLIC_V27_OPPONENT)
    parser.add_argument(
        "--games",
        type=int,
        default=8,
        help="seeds compared action by action, both seats scripted",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2,
        help="seeds replayed through the official engine per pairing; 0 skips",
    )
    parser.add_argument("--seed-start", type=int, default=90_001)
    parser.add_argument("--steps", type=int, default=MAX_TRANSITIONS)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _load_agent(path: Path) -> Any:
    """Import the agent file into a private module object.

    Never registered in `sys.modules`: each call must yield independent
    module-level state, because the agent's weed repair lives in a global.
    """
    spec = importlib.util.spec_from_file_location("kaggriculture_public_v27_audit", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import the public v27 agent from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observation(snapshot: dict[str, Any], player: int) -> dict[str, Any]:
    """Shape the native state into the observation the real agent reads."""
    return {
        "player": player,
        "step": snapshot["step"],
        "day": snapshot["day"],
        "hour": snapshot["hour"],
        "farms": snapshot["farms"],
        "private": snapshot["privates"][player],
        "market": snapshot["market"],
        "town": snapshot["town"],
    }


def _unit_value(command: Any, where: str) -> int:
    """Encode one unit command as its factor value, refusing what it cannot."""
    if not isinstance(command, (list, tuple)) or not command:
        raise ValueError(f"{where}: the agent emitted an empty unit command")
    operation = str(command[0])
    if operation in BARE_OPERATIONS:
        return int(UnitAction[operation])
    if operation == "PASS":
        return int(UnitAction.PASS)
    if operation in {"NORTH", "SOUTH", "EAST", "WEST", "DROP"}:
        return int(UnitAction[operation])
    if operation == "PICKUP":
        return int(UnitAction[f"PICKUP_{command[1]}_{int(command[2])}"])
    if operation == "PLACE":
        return int(UnitAction[f"PLACE_{command[1]}"])
    if operation == "PLANT":
        return int(UnitAction[f"PLANT_{command[1]}"])
    raise ValueError(f"{where}: no engine action encodes {list(command)!r}")


def _market_factors(order: Any, where: str) -> tuple[int, int]:
    if not isinstance(order, (list, tuple)) or not order:
        raise ValueError(f"{where}: the agent emitted an empty market order")
    operation = str(order[0])
    if operation in {"HIRE", "BUY_LAND"}:
        return int(MarketKind[operation]), 0
    quantity = int(order[2])
    if quantity not in _QUANTITY_INDEX:
        raise ValueError(f"{where}: quantity {quantity} is outside the engine's bins")
    item = str(order[1])
    if operation == "SELL":
        return int(MarketKind[f"SELL_{item}"]), _QUANTITY_INDEX[quantity]
    return int(MarketKind[f"{operation}_{item}"]), _QUANTITY_INDEX[quantity]


def _reference_factors(
    action: dict[str, Any], units: int, where: str
) -> tuple[list[int], list[int], list[int]]:
    """The factors the real agent's dict asks for, padded the way the engine reads it."""
    commands = [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]
    if len(commands) > units:
        raise ValueError(f"{where}: {len(commands)} unit commands for {units} active units")
    values = [
        _unit_value(command, f"{where} unit {index}") for index, command in enumerate(commands)
    ]
    values += [int(UnitAction.PASS)] * (units - len(values))
    orders = list(action.get("market") or [])
    if len(orders) > MAX_MARKET_ORDERS:
        raise ValueError(f"{where}: {len(orders)} market orders exceeds {MAX_MARKET_ORDERS}")
    kinds = [int(MarketKind.STOP)] * MAX_MARKET_ORDERS
    quantities = [0] * MAX_MARKET_ORDERS
    for index, order in enumerate(orders):
        kind, quantity = _market_factors(order, f"{where} order {index}")
        kinds[index] = kind
        quantities[index] = quantity
    return values, kinds, quantities


def _difference(
    native_units: np.ndarray,
    native_kinds: np.ndarray,
    native_quantities: np.ndarray,
    reference: tuple[list[int], list[int], list[int]],
    units: int,
) -> str | None:
    """Describe the first way the native action differs from the real agent's."""
    values, kinds, quantities = reference
    for unit in range(units):
        if int(native_units[unit]) != values[unit]:
            return (
                f"unit {unit}: native {UnitAction(int(native_units[unit])).name} "
                f"vs reference {UnitAction(values[unit]).name}"
            )
    # Slots past the live unit count are unreachable, so only the active prefix
    # is compared: the engine never asks a unit that does not exist.
    for slot in range(MAX_MARKET_ORDERS):
        native_kind = int(native_kinds[slot])
        if native_kind != kinds[slot]:
            return (
                f"market slot {slot}: native {MarketKind(native_kind).name} "
                f"vs reference {MarketKind(kinds[slot]).name}"
            )
        if native_kind == int(MarketKind.STOP):
            # The engine stops reading at the first STOP, so the padding behind
            # it is don't-care on both sides.
            break
        if int(native_quantities[slot]) != quantities[slot]:
            return (
                f"market slot {slot} quantity: native "
                f"{QUANTITY_BINS[int(native_quantities[slot])]} vs reference "
                f"{QUANTITY_BINS[quantities[slot]]}"
            )
    return None


def _compare_actions(
    module: Any,
    agents: Sequence[Any],
    *,
    games: int,
    seed_start: int,
    steps: int,
) -> dict[str, Any]:
    """Step a native wave with both seats scripted, checking every action.

    One imported reference module per game, because the reference keeps its weed
    repair in a module-level global keyed only by seat: a single import shared
    across a wave lets one game overwrite another's pending repair, which
    reports a divergence at a step where the port is in fact exact. Kaggle runs
    one episode per process, so per-game isolation is what faithfulness means.
    """
    seeds = np.arange(seed_start, seed_start + games, dtype=np.uint64)
    environment = module.BatchEnv(seeds)
    codes = np.full(games * PLAYERS, BUILTIN_CODES["scripted-v27"], dtype=np.uint8)
    compared = 0
    divergences: list[dict[str, Any]] = []
    for step in range(steps):
        snapshots = [json.loads(environment.snapshot_json(index, False)) for index in range(games)]
        # One call per step: the scripted agent advances its weed repair here.
        native = environment.builtin_actions(codes)
        units = np.asarray(native["unit_actions"], dtype=np.uint8)
        kinds = np.asarray(native["market_kinds"], dtype=np.uint8)
        quantities = np.asarray(native["market_quantities"], dtype=np.uint8)
        for index, snapshot in enumerate(snapshots):
            for player in range(PLAYERS):
                row = index * PLAYERS + player
                observation = _observation(snapshot, player)
                active = 1 + len(observation["farms"][player].get("hands") or [])
                where = f"seed {int(seeds[index])} step {step} seat {player}"
                reference = _reference_factors(
                    agents[index].agent(observation, _CONFIGURATION), active, where
                )
                reason = _difference(units[row], kinds[row], quantities[row], reference, active)
                compared += 1
                if reason is not None:
                    divergences.append({"where": where, "reason": reason})
        if divergences:
            break
        # Both seats are the reference agent here, whose emission is a submitted
        # dict: the engine owes it the interpreter's rules, not our policy mask.
        environment.step_factors(
            units.reshape(games, PLAYERS, -1),
            kinds.reshape(games, PLAYERS, -1),
            quantities.reshape(games, PLAYERS, -1),
            external=True,
        )
    return {"actions_compared": compared, "divergences": divergences}


#: The configuration the reference agent reads. `townCenterSellInterval` selects
#: its regime, so it must be the engine's own value or the port would be judged
#: against a branch it never takes.
_CONFIGURATION = {
    "episodeSteps": EPISODE_STEPS,
    "turnsPerDay": 24,
    "townShopSellInterval": 4,
    "townCenterSellInterval": 24,
    "townShopUnlockInterval": 3,
}


def _zero_inputs(games: int, module: Any) -> dict[str, np.ndarray]:
    """Sampling inputs for a wave whose every row is a built-in."""
    rows = games * PLAYERS
    return {
        "unit_logits": np.zeros((rows, module.MAX_UNITS, module.N_UNIT_ACTIONS), dtype=np.float32),
        "market_kind_logits": np.zeros(
            (rows, module.MAX_MARKET_ORDERS, module.N_MARKET_KINDS), dtype=np.float32
        ),
        "market_quantity_context": np.zeros((rows, module.MAX_MARKET_ORDERS, 1), dtype=np.float32),
        "quantity_kind_gate": np.zeros((1, module.N_MARKET_KINDS, 1), dtype=np.float32),
        "quantity_values": np.zeros((1, module.N_QUANTITIES, 1), dtype=np.float32),
        "quantity_bias": np.zeros(
            (1, module.N_MARKET_KINDS, module.N_QUANTITIES), dtype=np.float32
        ),
        "head_ids": np.zeros(rows, dtype=np.uint16),
        "unit_draws": np.zeros((rows, module.MAX_UNITS), dtype=np.float32),
        "market_kind_draws": np.zeros((rows, module.MAX_MARKET_ORDERS), dtype=np.float32),
        "market_quantity_draws": np.zeros((rows, module.MAX_MARKET_ORDERS), dtype=np.float32),
        "deterministic_rows": np.ones(rows, dtype=bool),
        "temperatures": np.ones(rows, dtype=np.float32),
    }


def _compare_episodes(
    module: Any,
    *,
    episodes: int,
    seed_start: int,
    agent_path: Path,
) -> list[dict[str, Any]]:
    """Play each pairing natively and through the official engine, comparing banks."""
    if not episodes:
        return []
    # The asymmetric pairings pin the seat index down; scripted against itself
    # puts both agents into one market, which is a different price path and the
    # only case where the sell-slot reordering is under real pressure.
    pairings = (
        ("scripted-v27", "pass"),
        ("pass", "scripted-v27"),
        ("scripted-v27", "scripted-v27"),
    )
    seeds = np.arange(seed_start, seed_start + episodes, dtype=np.uint64)
    official_names = {"scripted-v27": str(agent_path), "pass": "pass"}
    outcomes = []
    for pairing in pairings:
        environment = module.BatchEnv(seeds)
        sampled = environment.sample_buffers()
        inputs = _zero_inputs(episodes, module)
        codes = np.tile(
            np.array([BUILTIN_CODES[name] for name in pairing], dtype=np.uint8), episodes
        )
        for _ in range(MAX_TRANSITIONS):
            environment.sample_and_step_into(
                inputs["unit_logits"],
                inputs["market_kind_logits"],
                inputs["market_quantity_context"],
                inputs["quantity_kind_gate"],
                inputs["quantity_values"],
                inputs["quantity_bias"],
                inputs["head_ids"],
                inputs["unit_draws"],
                inputs["market_kind_draws"],
                inputs["market_quantity_draws"],
                inputs["deterministic_rows"],
                inputs["temperatures"],
                codes,
                sampled,
            )
        native_money = np.asarray(sampled["final_money"], dtype=np.float64)
        for index, seed in enumerate(seeds):
            official = make(
                "kaggriculture",
                configuration={"episodeSteps": EPISODE_STEPS, "seed": int(seed)},
                debug=False,
            )
            official.run([official_names[name] for name in pairing])
            outcomes.append(
                {
                    "pairing": list(pairing),
                    "seed": int(seed),
                    "native_money": [float(value) for value in native_money[index]],
                    "official_money": [
                        float(farm["money"]) for farm in official.state[0].observation.farms
                    ],
                }
            )
    return outcomes


def compare_v27_parity(
    *,
    agent_path: Path = PUBLIC_V27_OPPONENT,
    games: int = 1,
    episodes: int = 1,
    seed_start: int = 90_001,
    steps: int = MAX_TRANSITIONS,
    build: bool = False,
    release: bool = True,
) -> dict[str, Any]:
    """Run the digest-pinned action and final-bank comparisons."""
    if games < 1:
        raise ValueError("games must be positive")
    if episodes < 0:
        raise ValueError("episodes must be non-negative")
    if not 0 <= steps <= MAX_TRANSITIONS:
        raise ValueError(f"steps must be between 0 and {MAX_TRANSITIONS}")

    resolved_agent = agent_path.expanduser().resolve()
    if not resolved_agent.is_file():
        raise FileNotFoundError(f"public v27 agent is unavailable: {resolved_agent}")
    module = load_native(build=build, release=release)
    digest = hashlib.sha256(resolved_agent.read_bytes()).hexdigest()
    if digest != module.V27_SOURCE_SHA256:
        raise ValueError(
            "the native table was compiled from different bytes than the agent under "
            f"test: table {module.V27_SOURCE_SHA256} vs file {digest}. Regenerate with "
            "scripts/extract_v27_script.py."
        )

    agents = [_load_agent(resolved_agent) for _ in range(games)]
    action_result = _compare_actions(
        module,
        agents,
        games=games,
        seed_start=seed_start,
        steps=steps,
    )
    episode_result = _compare_episodes(
        module,
        episodes=episodes,
        seed_start=seed_start,
        agent_path=resolved_agent,
    )
    money_failures = [
        f"seed {row['seed']} {'/'.join(row['pairing'])}: native {row['native_money']} "
        f"vs official {row['official_money']}"
        for row in episode_result
        if row["native_money"] != row["official_money"]
    ]
    return {
        "agent": str(resolved_agent),
        "agent_sha256": digest,
        "scripted_steps": int(module.V27_STEPS),
        "actions_compared": action_result["actions_compared"],
        "divergences": action_result["divergences"],
        "episodes": episode_result,
        "money_failures": money_failures,
    }


def _parity_failures(report: dict[str, Any]) -> list[str]:
    return [f"{row['where']}: {row['reason']}" for row in report["divergences"]] + report[
        "money_failures"
    ]


def assert_v27_parity(
    *,
    agent_path: Path = PUBLIC_V27_OPPONENT,
    games: int = 1,
    episodes: int = 1,
    seed_start: int = 90_001,
    steps: int = MAX_TRANSITIONS,
    build: bool = False,
    release: bool = True,
) -> dict[str, Any]:
    """Assert exact scripted-v27 actions and final banks, returning the evidence report."""
    report = compare_v27_parity(
        agent_path=agent_path,
        games=games,
        episodes=episodes,
        seed_start=seed_start,
        steps=steps,
        build=build,
        release=release,
    )
    failures = _parity_failures(report)
    if failures:
        raise AssertionError("scripted v27 parity failed:\n  " + "\n  ".join(failures))
    return report


def main() -> int:
    args = parse_args()
    try:
        report = compare_v27_parity(
            agent_path=args.agent,
            games=args.games,
            episodes=args.episodes,
            seed_start=args.seed_start,
            steps=args.steps,
            build=True,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    print(
        f"compared {report['actions_compared']} actions over {args.games} seeds; "
        f"{len(report['episodes'])} official episodes"
    )
    failures = _parity_failures(report)
    if failures:
        raise SystemExit("scripted v27 parity failed:\n  " + "\n  ".join(failures))
    print("exact parity: 0 action divergences, every official bank matched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
