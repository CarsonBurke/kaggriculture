"""Project demonstrated engine actions into exact factored training targets.

The extractor turns a recorded engine action dict (``{"farmer", "hands",
"market"}``) into the factored representation the policy trains on, together
with the same sequential legality masks the sampler would have produced at
that state.

The projection target is the action the engine *executes*, not the bytes the
teacher emitted: the official interpreter ignores arguments beyond each
opcode's arity, clamps pickups to shed stock, silently no-ops commands whose
preconditions fail, fills market orders one unit at a time until resources
run out, and discards zero-fill orders. Each of those reductions is mirrored
here with the engine as the cited authority. Anything else — a state change
the engine would make that our factored space cannot represent, or a command
our legality mask forbids that the engine would execute — raises
:class:`DemonstrationError`. Silent substitution there would bake mask-model
divergence from the engine into the dataset — the highest-value failure mode
this pipeline can detect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from kaggriculture.actions import (
    _MOVE_DELTA,
    N_MARKET_KINDS,
    N_QUANTITIES,
    N_UNIT_ACTIONS,
    QUANTIFIED_MARKET_KINDS,
    MarketKind,
    UnitAction,
    _unit_inventory,
    _unit_position,
    apply_unit_shed_effect,
    apply_unit_tile_effect,
    compile_action,
    copy_tile_grid,
    unit_action_mask,
)
from kaggriculture.constants import (
    ANIMAL_STRUCTURE,
    BOARD_SIZE,
    CROP_FIRST_YIELD_DAY,
    MARKET_I0,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    PRODUCTS,
    QUANTITY_BINS,
    SHED_CAPACITY,
    fibonacci_hire_cost,
    shed_access_tiles,
)

# Package-internal reuse of the sampler's ledger machinery: the projection must
# evolve masks with byte-identical semantics to act_batch, so it shares the
# exact helpers instead of reimplementing them.
from kaggriculture.policy import (
    MarketLedger,
    _apply_ledger_order,
    _ledger_kind_mask,
    _ledger_quantity_mask,
)

_MOVE_NAMES = frozenset(("NORTH", "SOUTH", "EAST", "WEST"))
_SIMPLE_UNIT_NAMES = frozenset(
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
_PICKUP_MAX = {
    "WHEAT": 16,
    "FERTILIZER": 8,
    "GOOSE": 4,
    "COW": 4,
    "SHEEP": 4,
}
_BUY_SEED_KINDS = {
    "WHEAT": MarketKind.BUY_SEED_WHEAT,
    "CARROT": MarketKind.BUY_SEED_CARROT,
    "TOMATO": MarketKind.BUY_SEED_TOMATO,
    "STRAWBERRY": MarketKind.BUY_SEED_STRAWBERRY,
    "MELON": MarketKind.BUY_SEED_MELON,
}
_BUY_PRODUCT_KINDS = {
    "WHEAT": MarketKind.BUY_PRODUCT_WHEAT,
    "FERTILIZER": MarketKind.BUY_PRODUCT_FERTILIZER,
}
_BUY_ANIMAL_KINDS = {
    "GOOSE": MarketKind.BUY_ANIMAL_GOOSE,
    "COW": MarketKind.BUY_ANIMAL_COW,
    "SHEEP": MarketKind.BUY_ANIMAL_SHEEP,
}
_SELL_KINDS = {
    product: MarketKind(int(MarketKind.SELL_WHEAT) + index)
    for index, product in enumerate(PRODUCTS)
}


class DemonstrationError(ValueError):
    """A demonstrated action cannot be represented or is judged illegal.

    Raised instead of clamping: every occurrence is either a gap in the
    factored action space or a divergence between the Python legality model
    and the official engine, and both must surface during extraction rather
    than corrupt the dataset.
    """


@dataclass(frozen=True)
class ProjectedStep:
    """One state's factored targets plus the masks the sampler would have."""

    unit_actions: np.ndarray  # [MAX_UNITS] int8
    market_kinds: np.ndarray  # [MAX_MARKET_ORDERS] int8
    market_quantities: np.ndarray  # [MAX_MARKET_ORDERS] int8
    unit_masks: np.ndarray  # [MAX_UNITS, N_UNIT_ACTIONS] bool
    market_kind_masks: np.ndarray  # [MAX_MARKET_ORDERS, N_MARKET_KINDS] bool
    market_quantity_masks: np.ndarray  # [MAX_MARKET_ORDERS, N_QUANTITIES] bool
    unit_active: np.ndarray  # [MAX_UNITS] bool
    market_active: np.ndarray  # [MAX_MARKET_ORDERS] bool
    market_quantity_active: np.ndarray  # [MAX_MARKET_ORDERS] bool
    # The engine-executed reduction of the demonstrated action: arguments the
    # interpreter never reads are dropped and pickup quantities carry the
    # engine's shed clamp. `compile_action` on the factors must reproduce this
    # dict exactly (verify_round_trip).
    canonical_action: dict[str, Any]


def _canonical_unit_command(command: Any) -> list[Any]:
    """Reduce a demonstrated unit command to what the engine actually reads.

    The interpreter ignores arguments beyond each opcode's arity (v27 emits
    decorated forms like ``["FEED", "WHEAT"]``), so canonicalization — not
    exact-list comparison — is the faithful equality for round trips.
    """
    if not isinstance(command, (list, tuple)) or not command:
        raise DemonstrationError(f"unparseable unit command: {command!r}")
    opcode = str(command[0])
    if opcode == "PASS" or opcode in _MOVE_NAMES or opcode in _SIMPLE_UNIT_NAMES:
        return [opcode]
    if opcode == "DROP":
        return ["DROP"]
    if opcode in ("PLANT", "PLACE"):
        if len(command) < 2:
            raise DemonstrationError(f"{opcode} without an argument: {command!r}")
        if opcode == "PLACE" and len(command) >= 3 and int(command[2]) != 1:
            # The engine's PLACE doubles as a shed deposit of N items; the
            # factored space only represents the animal-placement/deposit-1
            # form, so a larger deposit is a representability gap.
            raise DemonstrationError(f"unrepresentable PLACE deposit quantity: {command!r}")
        return [opcode, str(command[1])]
    if opcode == "PICKUP":
        if len(command) < 2:
            raise DemonstrationError(f"PICKUP without an item: {command!r}")
        quantity = int(command[2]) if len(command) >= 3 else 1
        return ["PICKUP", str(command[1]), quantity]
    raise DemonstrationError(f"unknown unit opcode: {command!r}")


def _parse_unit_command(
    command: Any, shed_available: dict[str, int]
) -> tuple[UnitAction, list[Any]]:
    """Map a demonstrated command to its factored variant and executed form.

    Returns the selected :class:`UnitAction` plus the command the engine
    actually executes at this ledger state, which is what ``compile_action``
    must reproduce. The only stateful reduction is PICKUP's shed clamp
    (``n = min(n, available)``, no-op when nothing is available) — every other
    reduction is pure argument arity.
    """
    canonical = _canonical_unit_command(command)
    opcode = str(canonical[0])
    if opcode == "PASS":
        return UnitAction.PASS, canonical
    if opcode in _MOVE_NAMES or opcode in _SIMPLE_UNIT_NAMES:
        return UnitAction[opcode], canonical
    if opcode == "DROP":
        return UnitAction.DROP, canonical
    if opcode in ("PLANT", "PLACE"):
        try:
            return UnitAction[f"{opcode}_{canonical[1]}"], canonical
        except KeyError:
            raise DemonstrationError(f"unknown {opcode} target in {command!r}") from None
    item, quantity = str(canonical[1]), int(canonical[2])
    if quantity <= 0:
        # Engine: `if n <= 0: return` — a demonstrated non-positive pickup
        # executes as nothing at all.
        return UnitAction.PASS, ["PASS"]
    maximum = _PICKUP_MAX.get(item)
    if maximum is None:
        raise DemonstrationError(f"unknown pickup item in {command!r}")
    executed = min(quantity, int(shed_available.get(item, 0) or 0))
    if executed <= 0:
        # Engine clamps to shed stock and no-ops on an empty shed; the
        # executed behavior is exactly PASS.
        return UnitAction.PASS, ["PASS"]
    if executed > maximum:
        raise DemonstrationError(
            f"executed pickup of {executed} {item} is outside the factored "
            f"action space (1..{maximum})"
        )
    # compile_action emits min(variant, shed); with variant == executed <= shed
    # the emitted command matches the engine-executed one exactly.
    return UnitAction[f"PICKUP_{item}_{executed}"], ["PICKUP", item, executed]


def _canonical_market_order(order: Any) -> list[Any]:
    """Reduce a demonstrated market order to what the engine actually reads."""
    if not isinstance(order, (list, tuple)) or not order:
        raise DemonstrationError(f"unparseable market order: {order!r}")
    opcode = str(order[0])
    if opcode in ("HIRE", "BUY_LAND"):
        return [opcode]
    if opcode in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"):
        if len(order) < 3:
            raise DemonstrationError(f"market order without item/quantity: {order!r}")
        return [opcode, str(order[1]), int(order[2])]
    raise DemonstrationError(f"unknown market order: {order!r}")


def _parse_market_order(order: Any) -> tuple[MarketKind, int]:
    canonical = _canonical_market_order(order)
    opcode = str(canonical[0])
    if opcode == "HIRE":
        return MarketKind.HIRE, 0
    if opcode == "BUY_LAND":
        return MarketKind.BUY_LAND, 0
    item, quantity = str(canonical[1]), int(canonical[2])
    if quantity < 1:
        # The engine drops n <= 0 orders as malformed; surface rather than
        # guess what the teacher meant.
        raise DemonstrationError(f"non-positive market quantity: {order!r}")
    tables = {
        "BUY_SEED": _BUY_SEED_KINDS,
        "BUY_PRODUCT": _BUY_PRODUCT_KINDS,
        "BUY_ANIMAL": _BUY_ANIMAL_KINDS,
        "SELL": _SELL_KINDS,
    }
    table = tables[opcode]
    if item not in table:
        raise DemonstrationError(f"unknown market order: {order!r}")
    # Demands above the largest bin are legitimate engine over-asks (the
    # engine fills per-unit until resources run out); the projection clamps
    # them to the mask's affordability bound below.
    return table[item], min(quantity, QUANTITY_BINS[-1]) - 1


def _engine_would_execute(
    observation: dict[str, Any],
    unit_index: int,
    selected: UnitAction,
    tiles: list[list[Any]],
    remaining_seeds: dict[str, int],
    remaining_shed: dict[str, int],
) -> bool:
    """Would the official interpreter change any state executing this command?

    Mirrors the preconditions of ``_apply_unit_action`` in the official
    kaggriculture interpreter, evaluated against the sequential within-turn
    ledger state (tiles/seeds/shed evolve as earlier units act; positions and
    per-unit inventories do not, because each unit acts exactly once). This
    arbitrates demonstrated commands our legality mask forbids: an engine
    no-op is behaviorally PASS, while an engine state change means the mask
    model has a real divergence (or the factored space a gap) and extraction
    must abort.
    """
    player = int(observation.get("player", 0) or 0)
    farm = (observation.get("farms") or [])[player]
    position = _unit_position(farm, unit_index)
    if position is None:
        return False
    x, y = position
    board_size = len(tiles) or BOARD_SIZE
    inventory = _unit_inventory(observation.get("private") or {}, unit_index)
    at_shed = position in shed_access_tiles(board_size)

    if selected == UnitAction.PASS:
        return False
    if selected in _MOVE_DELTA:
        dx, dy = _MOVE_DELTA[selected]
        # The engine allows moves onto LOCKED tiles; only the board edge blocks.
        return 0 <= x + dx < board_size and 0 <= y + dy < board_size
    if selected == UnitAction.DROP:
        # A drop with a full shed still destroys the unit's inventory, so any
        # carried item makes this a real state change.
        return at_shed and any(int(value or 0) > 0 for value in inventory.values())

    name = selected.name
    if name.startswith("PICKUP_"):
        item = name[len("PICKUP_") :].rsplit("_", 1)[0]
        return at_shed and int(remaining_shed.get(item, 0) or 0) > 0

    tile = tiles[y][x]
    if name.startswith("PLACE_"):
        animal = name[len("PLACE_") :]
        holds_animal = int(inventory.get(animal, 0) or 0) > 0
        installs = (
            isinstance(tile, dict)
            and tile.get("kind") == ANIMAL_STRUCTURE[animal]
            and "animal" not in tile
        )
        if installs and holds_animal:
            return True
        # Otherwise PLACE falls through to the engine's shed-deposit path.
        shed_room = SHED_CAPACITY - sum(int(value or 0) for value in remaining_shed.values())
        return at_shed and holds_animal and shed_room > 0

    # Everything below mutates the standing tile, which must be owned.
    if tile == "LOCKED":
        return False
    if name.startswith("PLANT_"):
        crop = name[len("PLANT_") :]
        return tile is None and int(remaining_seeds.get(crop, 0) or 0) > 0
    if selected in (UnitAction.BUILD_COOP, UnitAction.BUILD_PASTURE):
        return tile is None
    if selected == UnitAction.DIG:
        return tile is not None and not (isinstance(tile, dict) and "animal" in tile)
    if not isinstance(tile, dict):
        return False
    if selected == UnitAction.WATER:
        return tile.get("kind") == "PLANT" and not bool(tile.get("watered_today", False))
    if selected == UnitAction.HARVEST:
        if int(tile.get("yield_units", 0) or 0) <= 0:
            return False
        if tile.get("kind") == "PLANT":
            day = int(observation.get("day", 0) or 0)
            age = day - int(tile.get("planted_day", day) or 0)
            return age >= CROP_FIRST_YIELD_DAY.get(tile.get("crop"), 10**9)
        return "animal" in tile
    if selected == UnitAction.FERTILIZE:
        # The engine consumes the fertilizer even when the tile is already
        # fertilized through day+2, so possession alone makes this real.
        return tile.get("kind") == "PLANT" and int(inventory.get("FERTILIZER", 0) or 0) > 0
    if selected == UnitAction.FEED:
        return (
            "animal" in tile
            and not bool(tile.get("fed_today", False))
            and int(inventory.get("WHEAT", 0) or 0) > 0
        )
    if selected == UnitAction.COLLECT_FERTILIZER:
        return "animal" in tile and bool(tile.get("fertilizer_available", False))
    if selected == UnitAction.CARE:
        return "animal" in tile and not bool(tile.get("cared_today", False))
    raise DemonstrationError(f"no engine-execution model for {selected.name}")


def project_demonstration(observation: dict[str, Any], action: dict[str, Any]) -> ProjectedStep:
    """Project one demonstrated engine action through the sequential ledger.

    Runs the exact mask evolution the sampler uses (`act_batch` semantics),
    substituting the demonstrated selections reduced to their engine-executed
    form, and raises :class:`DemonstrationError` the moment the engine would
    make a state change our factored space or legality model cannot express.
    On success the factors round-trip through `compile_action` to the
    projection's canonical action — verified by :func:`verify_round_trip`,
    which extraction must always call.
    """
    player = int(observation.get("player", 0) or 0)
    farm = (observation.get("farms") or [])[player]
    hands = farm.get("hands") or []
    unit_count = min(MAX_UNITS, 1 + len(hands))
    commands = [action.get("farmer") or ["PASS"]]
    demonstrated_hands = list(action.get("hands") or [])
    if len(demonstrated_hands) > len(hands):
        raise DemonstrationError(
            f"action lists {len(demonstrated_hands)} hands but the farm has {len(hands)}"
        )
    # A short hand list means the trailing units did nothing this turn.
    demonstrated_hands.extend(["PASS"] for _ in range(len(hands) - len(demonstrated_hands)))
    commands.extend(demonstrated_hands)

    unit_actions = np.full(MAX_UNITS, int(UnitAction.PASS), dtype=np.int8)
    unit_masks = np.zeros((MAX_UNITS, N_UNIT_ACTIONS), dtype=np.bool_)
    unit_active = np.zeros(MAX_UNITS, dtype=np.bool_)
    remaining_seeds = dict((observation.get("private") or {}).get("seeds") or {})
    remaining_shed = dict((observation.get("private") or {}).get("shed") or {})
    tiles = copy_tile_grid(farm.get("tiles") or [])
    canonical_commands: list[list[Any]] = []
    for unit in range(MAX_UNITS):
        if unit >= unit_count:
            unit_masks[unit, UnitAction.PASS] = True
            continue
        unit_active[unit] = True
        unit_masks[unit] = unit_action_mask(
            observation, unit, remaining_seeds, remaining_shed, tiles
        )
        selected, canonical = _parse_unit_command(commands[unit], remaining_shed)
        if not unit_masks[unit, selected]:
            if _engine_would_execute(
                observation, unit, selected, tiles, remaining_seeds, remaining_shed
            ):
                raise DemonstrationError(
                    f"unit {unit} demonstrated {selected.name}, which our legality "
                    "model forbids but the engine would execute — mask divergence"
                )
            # The engine silently no-ops this command at this ledger state
            # (e.g. WATER on an empty tile from v27's open-loop trace), so the
            # executed behavior is exactly PASS.
            selected, canonical = UnitAction.PASS, ["PASS"]
        canonical_commands.append(canonical)
        unit_actions[unit] = int(selected)
        crop = selected.name.removeprefix("PLANT_")
        if crop != selected.name:
            remaining_seeds[crop] = int(remaining_seeds.get(crop, 0) or 0) - 1
        apply_unit_shed_effect(observation, unit, selected, remaining_shed, tiles)
        apply_unit_tile_effect(observation, unit, selected, tiles)

    # The engine truncates each queue to maxMarketOrdersPerTurn before any
    # execution, so trailing extras are never read.
    orders = list(action.get("market") or [])[:MAX_MARKET_ORDERS]
    market_inventory = (observation.get("market") or {}).get("inventory") or {}
    ledger = MarketLedger(
        money=float(farm.get("money", 0) or 0),
        shed=remaining_shed,
        hires=int(farm.get("hires_today", 0) or 0),
        extra_land=max(0, len(farm.get("unlocked_quadrants") or []) - 1),
        inventory={item: int(market_inventory.get(item, MARKET_I0)) for item in PRODUCTS},
    )
    canonical_orders: list[list[Any]] = []
    market_kinds = np.full(MAX_MARKET_ORDERS, int(MarketKind.STOP), dtype=np.int8)
    market_quantities = np.zeros(MAX_MARKET_ORDERS, dtype=np.int8)
    kind_masks = np.zeros((MAX_MARKET_ORDERS, N_MARKET_KINDS), dtype=np.bool_)
    quantity_masks = np.zeros((MAX_MARKET_ORDERS, N_QUANTITIES), dtype=np.bool_)
    market_active = np.zeros(MAX_MARKET_ORDERS, dtype=np.bool_)
    quantity_active = np.zeros(MAX_MARKET_ORDERS, dtype=np.bool_)
    for order in orders:
        canonical_order = _canonical_market_order(order)
        kind, quantity_index = _parse_market_order(order)
        slot = len(canonical_orders)
        kind_mask_now = _ledger_kind_mask(observation, ledger)
        if not kind_mask_now[kind]:
            # The ledger kind mask is exactly the engine's first-per-unit-commit
            # predicate (SELL: stock > 0; BUY_*: money covers the first quote,
            # shed room for goods; BUY_LAND: land left and affordable), except
            # that HIRE additionally carries the factored 16-unit cap. A
            # forbidden kind therefore means the engine fills zero units and
            # discards the order — a behavioral no-op the projection drops —
            # unless the engine would actually hire past our unit cap.
            if kind == MarketKind.HIRE and ledger.money >= fibonacci_hire_cost(ledger.hires):
                raise DemonstrationError(
                    "demonstrated HIRE beyond the factored 16-unit cap — representability gap"
                )
            continue
        market_active[slot] = True
        kind_masks[slot] = kind_mask_now
        market_kinds[slot] = int(kind)
        quantity_masks[slot] = _ledger_quantity_mask(observation, kind, ledger)
        if kind in QUANTIFIED_MARKET_KINDS:
            quantity_active[slot] = True
            if not quantity_masks[slot, quantity_index]:
                # The engine fills quantified orders one unit at a time and
                # simply stops when resources run out, so a demand above our
                # ledger's affordability bound executes as the bound itself
                # (verified against real games: BUY_SEED MELON x7 with money
                # for 4 fills exactly 4). Project the executed fill.
                affordable = np.flatnonzero(quantity_masks[slot, :quantity_index])
                if affordable.size == 0:
                    raise DemonstrationError(
                        f"market slot {slot} demonstrated {kind.name} x"
                        f"{QUANTITY_BINS[quantity_index]} but our ledger affords "
                        "no quantity at all — mask divergence"
                    )
                quantity_index = int(affordable[-1])
            market_quantities[slot] = quantity_index
            canonical_order = [*canonical_order[:2], int(QUANTITY_BINS[quantity_index])]
        canonical_orders.append(canonical_order)
        _apply_ledger_order(observation, kind, QUANTITY_BINS[quantity_index], ledger)

    if len(canonical_orders) < MAX_MARKET_ORDERS:
        # The projected turn ends here; choosing STOP is itself a trained
        # decision, exactly as in the sampler.
        stop_slot = len(canonical_orders)
        market_active[stop_slot] = True
        kind_masks[stop_slot] = _ledger_kind_mask(observation, ledger)
        if not kind_masks[stop_slot, MarketKind.STOP]:
            raise DemonstrationError("STOP is masked out, which should be impossible")
        quantity_masks[stop_slot, 0] = True
        for slot in range(stop_slot + 1, MAX_MARKET_ORDERS):
            kind_masks[slot, MarketKind.STOP] = True
            quantity_masks[slot, 0] = True

    return ProjectedStep(
        unit_actions=unit_actions,
        market_kinds=market_kinds,
        market_quantities=market_quantities,
        unit_masks=unit_masks,
        market_kind_masks=kind_masks,
        market_quantity_masks=quantity_masks,
        unit_active=unit_active,
        market_active=market_active,
        market_quantity_active=quantity_active,
        canonical_action={
            "farmer": canonical_commands[0],
            "hands": canonical_commands[1:],
            "market": canonical_orders,
        },
    )


def _normalized_action(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "farmer": [_normalize_scalar(part) for part in (action.get("farmer") or ["PASS"])],
        "hands": [
            [_normalize_scalar(part) for part in command] for command in (action.get("hands") or [])
        ],
        "market": [
            [_normalize_scalar(part) for part in order] for order in (action.get("market") or [])
        ],
    }


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def verify_round_trip(
    observation: dict[str, Any],
    action: dict[str, Any],
    projected: ProjectedStep,
) -> None:
    """Require the projected factors to recompile to the engine-executed action.

    ``compile_action`` re-runs the full sequential legality ledger from the raw
    observation, so exact equality against the projection's canonical action —
    the demonstrated dict reduced to what the interpreter reads and executes —
    proves the parse, the mask evolution, and the engine-facing compiler all
    agree on this state. The raw demonstrated dict appears in the error for
    diagnosis; it may legitimately differ from the canonical form only by
    ignored arguments and engine-clamped pickup quantities.
    """
    compiled = compile_action(
        observation,
        projected.unit_actions.astype(np.int64),
        projected.market_kinds.astype(np.int64),
        projected.market_quantities.astype(np.int64),
    )
    canonical = _normalized_action(projected.canonical_action)
    hand_count = len(compiled["hands"])
    if len(canonical["hands"]) != hand_count:
        raise DemonstrationError(
            f"projection produced {len(canonical['hands'])} hand commands but "
            f"compile_action emitted {hand_count}"
        )
    if _normalized_action(compiled) != canonical:
        raise DemonstrationError(
            "round trip diverged: "
            f"compiled {_normalized_action(compiled)!r} != canonical {canonical!r} "
            f"(demonstrated {_normalized_action(action)!r})"
        )
