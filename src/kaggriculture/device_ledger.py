"""Exact tensor policy ledger for turn-local causal decoding.

This is the *policy mask* contract, not a second game simulator. Every unit
acts once, so only shared seeds/shed and tiles under still-pending units mutate.
Positions, held inventories and insertion order are immutable turn-start inputs.
The market ledger intentionally excludes the simultaneous opponent's orders,
matching Rust's policy sampler; the native engine still settles the joint turn.

All decision methods are tensor-only and compile together with a GPU decoder.
Native integer quote tables avoid approximating the engine's f64 pricing rules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from torch import Tensor

from kaggriculture.constants import CROPS, MARKET_PARAMS, PRIVATE_ITEMS, PRODUCTS

POLICY_LEDGER_SCHEMA_VERSION = 1
POLICY_LEDGER_WIDTH = 593
# At most 2 players * 10 orders * 100 units change inventory each step.
# Eight shops can consume at most 16 units/product/step, plus one town unit.
# These bounds cover 720 default steps, with another 1000 units in each
# direction for all hypothetical quantities examined within one policy turn.
PRICE_MINIMUM = -1_444_000
PRICE_MAXIMUM = 1_452_000
PREFIX_INVENTORY_MARGIN = 1000


@cache
def _cached_device_ledger(device: torch.device) -> DeviceLedger:
    bundled = Path(__file__).with_name("policy-market-prices.npz")
    if bundled.is_file():
        with np.load(bundled, allow_pickle=False) as archive:
            if set(archive.files) != {"schema_version", "minimum", "prices"}:
                raise ValueError("invalid bundled policy price table fields")
            prices = archive["prices"]
            if (
                int(archive["schema_version"]) != POLICY_LEDGER_SCHEMA_VERSION
                or int(archive["minimum"]) != PRICE_MINIMUM
                or prices.dtype != np.int32
                or prices.shape != (9, PRICE_MAXIMUM - PRICE_MINIMUM + 1)
                or (prices < 1).any()
            ):
                raise ValueError("invalid bundled policy price table contract")
        return DeviceLedger(torch.as_tensor(prices, device=device), PRICE_MINIMUM)
    return DeviceLedger.from_native(device)


def get_device_ledger(device: torch.device | str) -> DeviceLedger:
    """One immutable exact native quote table per device, built before capture."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return _cached_device_ledger(device)


def _integer_cumsum(values: Tensor) -> Tensor:
    if values.is_cuda:
        from kaggriculture.device_ledger_kernels import ledger_cumsum

        return ledger_cumsum(values)
    return values.cumsum(-1)


class PolicyLedger(NamedTuple):
    day: Tensor
    money: Tensor
    hires: Tensor
    original_hires: Tensor
    units: Tensor
    land: Tensor
    capacity: Tensor
    hire_multiplier: Tensor
    seeds: Tensor
    shed: Tensor
    market: Tensor
    positions: Tensor
    inventories: Tensor
    inventory_order: Tensor
    tiles: Tensor
    market_active: Tensor


class MarketQuote(NamedTuple):
    kind: Tensor
    mask: Tensor
    money_delta: Tensor
    market_delta: Tensor
    item: Tensor


class ReplayMasks(NamedTuple):
    unit_masks: Tensor
    market_kind_masks: Tensor
    market_quantity_masks: Tensor
    unit_active: Tensor
    market_active: Tensor
    market_quantity_active: Tensor


def initial_ledger(packed: Tensor) -> PolicyLedger:
    """View one exact native/observation batch; no copies or host synchronization."""
    if packed.ndim != 2 or packed.shape[1] != POLICY_LEDGER_WIDTH or packed.dtype != torch.int64:
        raise ValueError("policy ledger must be int64 [batch, 593]")
    batch = packed.shape[0]
    return PolicyLedger(
        packed[:, 0],
        packed[:, 1],
        packed[:, 2],
        packed[:, 2],
        packed[:, 3],
        packed[:, 4],
        packed[:, 5],
        packed[:, 6],
        packed[:, 7:12],
        packed[:, 12:24],
        packed[:, 24:33],
        packed[:, 33:65].reshape(batch, 16, 2),
        packed[:, 65:257].reshape(batch, 16, 12),
        packed[:, 257:449].reshape(batch, 16, 12),
        packed[:, 449:593].reshape(batch, 16, 9),
        torch.ones_like(packed[:, 0], dtype=torch.bool),
    )


def validate_packed(packed: np.ndarray, minimum: int, maximum: int) -> None:
    """Validate at the host staging boundary, never inside a decoding graph.

    The table margin guarantees every lookup in all ten market decisions fits,
    including candidate quantities the selected action never executes.
    """
    if packed.ndim != 2 or packed.shape[1] != POLICY_LEDGER_WIDTH or packed.dtype != np.int64:
        raise ValueError("policy ledger must be int64 [batch, 593]")
    if (packed[:, 24:33] < minimum + PREFIX_INVENTORY_MARGIN).any() or (
        packed[:, 24:33] > maximum - PREFIX_INVENTORY_MARGIN
    ).any():
        raise ValueError("market inventory lies outside the exact price table's prefix bounds")
    if (packed[:, [0, 1, 2, 3, 4, 5, 6]] < 0).any():
        raise ValueError("negative policy ledger scalar")
    if (packed[:, 3] > 16).any() or (packed[:, 4] > 3).any() or (packed[:, 6] < 1).any():
        raise ValueError("unsupported unit count, land or hiring multiplier")
    if (packed[:, 33:65] < 0).any() or (packed[:, 33:65] >= 10).any():
        raise ValueError("unit position lies outside the board")
    if (packed[:, 257:449] < 0).any() or (packed[:, 257:449] > 12).any():
        raise ValueError("invalid inventory insertion-order index")
    if (packed[:, 7:24] < 0).any() or (packed[:, 65:257] < 0).any():
        raise ValueError("negative private stock")
    tiles = packed[:, 449:].reshape(-1, 16, 9)
    if (
        (tiles[..., 0] < 0).any()
        or (tiles[..., 0] > 5).any()
        or ((tiles[..., 1] < 0) | (tiles[..., 1] > 4)).any()
    ):
        raise ValueError("invalid local tile kind/species")


def validate_replay_contract(
    packed: np.ndarray,
    unit_actions: np.ndarray,
    market_kinds: np.ndarray,
    market_quantities: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> None:
    """Verify a BC episode against exact rules before writing its encoded cache.

    This executes integer policy rules only, never a neural model. Restricting
    the native quote table to this episode's reachable prefix inventory avoids
    a full-game quote table in every preprocessing worker. Mismatched labels or
    support fail rather than silently changing the demonstrations.
    """
    validate_packed(packed, PRICE_MINIMUM, PRICE_MAXIMUM)
    if not len(packed):
        raise ValueError("empty causal demonstration episode")
    factors = (unit_actions, market_kinds, market_quantities)
    for name, values, slots, support in zip(
        ("unit_actions", "market_kinds", "market_quantities"),
        factors,
        (16, 10, 10),
        (68, 22, 100),
        strict=True,
    ):
        if (
            values.shape != (len(packed), slots)
            or not np.issubdtype(values.dtype, np.integer)
            or (values < 0).any()
            or (values >= support).any()
        ):
            raise ValueError(f"causal demonstration {name} has invalid shape or factor index")
    minimum = int(packed[:, 24:33].min()) - PREFIX_INVENTORY_MARGIN
    maximum = int(packed[:, 24:33].max()) + PREFIX_INVENTORY_MARGIN
    rules = DeviceLedger.from_native("cpu", minimum=minimum, maximum=maximum)
    with torch.no_grad():
        replay = rules.replay_masks(
            torch.from_numpy(packed),
            *(torch.as_tensor(values.astype(np.int64, copy=False)) for values in factors),
        )
    actual = {name: value.numpy() for name, value in zip(replay._fields, replay, strict=True)}
    for name, values in actual.items():
        expected = np.asarray(masks[name])
        if expected.shape != values.shape:
            raise ValueError(f"causal demonstration {name} shape disagrees with exact policy")
        different = expected != values
        if different.any():
            index = tuple(np.argwhere(different)[0])
            raise ValueError(f"causal demonstration {name} differs from exact policy at {index}")
    for values, name, active in zip(
        factors,
        ("unit_masks", "market_kind_masks", "market_quantity_masks"),
        ("unit_active", "market_active", "market_quantity_active"),
        strict=True,
    ):
        legal = np.take_along_axis(actual[name], values[..., None].astype(np.int64), -1)[..., 0]
        illegal = actual[active] & ~legal
        if illegal.any():
            index = tuple(np.argwhere(illegal)[0])
            raise ValueError(f"causal demonstration illegal selected action in {name} at {index}")


class DeviceLedger:
    """Shared immutable rules and quotes; reuse across all actor replicas.

    This object has no trainable parameters. Do not copy the quote table into
    each frozen actor or include it in checkpoints. Construct it once outside
    CUDA graph capture and pass/reuse it for collection, replay and inference.
    """

    def __deepcopy__(self, memo: dict) -> DeviceLedger:
        # Frozen policy replicas own weights, never another immutable price table.
        memo[id(self)] = self
        return self

    def __init__(self, prices: Tensor, minimum: int) -> None:
        if prices.ndim != 2 or prices.shape[0] != 9 or prices.dtype != torch.int32:
            raise ValueError("native price table must be int32 [9, inventory range]")
        self.prices = prices
        self.minimum = minimum
        self.maximum = minimum + prices.shape[1] - 1
        device = prices.device

        def ints(values):
            return torch.tensor(values, device=device, dtype=torch.int64)

        self.action_ids = torch.arange(68, device=device)
        self.items = torch.arange(12, device=device)
        self.crop_ids = torch.arange(5, device=device)
        self.quantities = torch.arange(1, 101, device=device)
        self.seed_cost = ints([10, 20, 50, 100, 80])
        self.animal_cost = ints([300, 400, 500])
        self.land_cost = ints([1000, 2000, 4000, 0])
        self.first_yield = ints([2, 2, 8, 10, 10])
        self.max_yield_day = ints([4, 3, 8, 10, 12])
        self.max_yield = ints([6, 4, 4, 4, 6])
        pickup_item = [0] * 68
        pickup_count = [0] * 68
        for first, last, item in ((6, 22, 0), (22, 30, 8), (30, 34, 9), (34, 38, 10), (38, 42, 11)):
            for action in range(first, last):
                pickup_item[action] = item
                pickup_count[action] = action - first + 1
        self.pickup_item = ints(pickup_item)
        self.pickup_count = ints(pickup_count)
        # fib(0)=fib(1)=1, saturating to i64::MAX exactly as the engine does.
        maximum = torch.iinfo(torch.int64).max
        fib = [1, 1]
        while fib[-1] < maximum:
            fib.append(min(maximum, fib[-1] + fib[-2]))
        self.fibonacci = ints(fib)

    @classmethod
    def from_native(
        cls,
        device: torch.device | str,
        *,
        minimum: int = PRICE_MINIMUM,
        maximum: int = PRICE_MAXIMUM,
    ) -> DeviceLedger:
        from kaggriculture.rust_env import load_native

        native = load_native()
        if native.POLICY_LEDGER_SCHEMA_VERSION != POLICY_LEDGER_SCHEMA_VERSION:
            raise ValueError("native policy ledger schema mismatch")
        prices = native.BatchEnv.policy_market_prices(minimum, maximum)
        return cls(torch.as_tensor(prices, device=device), minimum)

    def _hire_cost(self, state: PolicyLedger) -> Tensor:
        # Beyond the saturation index the exact native sequence remains MAX.
        fib = self.fibonacci[state.hires.clamp_max(self.fibonacci.shape[0] - 1)]
        maximum = torch.iinfo(torch.int64).max
        saturated = fib > torch.div(maximum, state.hire_multiplier, rounding_mode="floor")
        return torch.where(saturated, maximum, fib * state.hire_multiplier)

    def unit_mask(self, state: PolicyLedger, unit: int) -> Tensor:
        a = self.action_ids[None]
        x, y = state.positions[:, unit].unbind(-1)
        tile = state.tiles[:, unit]
        kind, species, animal, origin, stock, watered, cared, fertilizer, until = tile.unbind(-1)
        held = state.inventories[:, unit]
        at_shed = ((x == 4) | (x == 5)) & ((y == 4) | (y == 5))
        room = state.capacity - state.shed.sum(-1)
        mask = (a == 0) | ((a == 1) & (y[:, None] > 0)) | ((a == 2) & (y[:, None] < 9))
        mask = mask | ((a == 3) & (x[:, None] < 9)) | ((a == 4) & (x[:, None] > 0))
        mask = mask | ((a == 5) & (at_shed & (held > 0).any(-1))[:, None])
        mask = mask | (
            (a >= 6)
            & (a < 42)
            & at_shed[:, None]
            & (state.shed[:, self.pickup_item] >= self.pickup_count)
        )
        for action, item, structure in ((42, 9, 4), (43, 10, 5), (44, 11, 5)):
            allowed = (held[:, item] > 0) & (
                ((kind == structure) & (animal == 0)) | (at_shed & (room > 0))
            )
            mask = mask | ((a == action) & allowed[:, None])
        for crop in range(5):
            mask = mask | ((a == 45 + crop) & ((kind == 0) & (state.seeds[:, crop] > 0))[:, None])
        harvest = (stock > 0) & (
            ((kind == 3) & ((state.day - origin).clamp_min(0) >= self.first_yield[species]))
            | (animal != 0)
        )
        allowed = (
            (50, (kind == 3) & (watered == 0)),
            (51, harvest),
            (52, (kind == 3) & (held[:, 8] > 0) & (until < state.day + 2)),
            (53, (kind != 0) & (kind != 1) & (animal == 0)),
            (54, kind == 0),
            (55, kind == 0),
            (56, (animal != 0) & (watered == 0) & (held[:, 0] > 0)),
            (57, (animal != 0) & (fertilizer != 0)),
            (58, (animal != 0) & (cared == 0)),
        )
        for action, valid in allowed:
            mask = mask | ((a == action) & valid[:, None])
        for item in range(9):
            mask = mask | ((a == 59 + item) & (at_shed & (room > 0) & (held[:, item] > 0))[:, None])
        return torch.where((unit < state.units)[:, None], mask, a == 0)

    def apply_unit(self, state: PolicyLedger, unit: int, action: Tensor) -> PolicyLedger:
        """Apply a selected legal action; immutable per-unit inputs stay at turn start."""
        active = unit < state.units
        action = torch.where(active, action, 0)
        tile = state.tiles[:, unit]
        kind, species, animal, origin, stock, _watered, _cared, _fertilizer, until = tile.unbind(-1)
        held = state.inventories[:, unit]
        position = state.positions[:, unit]
        at_shed = ((position == 4) | (position == 5)).all(-1)
        room = (state.capacity - state.shed.sum(-1)).clamp_min(0)
        pickup = (action >= 6) & (action < 42)
        item = self.pickup_item[action]
        shed = (
            state.shed
            - (self.items[None] == item[:, None])
            * torch.where(pickup, self.pickup_count[action], 0)[:, None]
        )
        # DROP allocates limited room in the original dictionary insertion order.
        order = state.inventory_order[:, unit]
        padded = torch.cat((held, torch.zeros_like(held[:, :1])), dim=1)
        ordered = padded.gather(1, order)
        preceding = _integer_cumsum(ordered) - ordered
        deposited = torch.minimum(ordered, (room[:, None] - preceding).clamp_min(0))
        ordered_delta = torch.zeros_like(padded).scatter_add(1, order, deposited)[:, :12]
        shed = shed + torch.where((action == 5)[:, None], ordered_delta, 0)
        animal_action = (action >= 42) & (action <= 44)
        animal_species = (action - 42).clamp(0, 2)
        structure = torch.where(animal_species == 0, 4, 5)
        installs = animal_action & (kind == structure) & (animal == 0)
        deposited_animal = animal_action & ~installs & at_shed & (room > 0)
        shed = shed + (
            (self.items[None] == (9 + animal_species)[:, None]) & deposited_animal[:, None]
        )
        product_action = (action >= 59) & (action <= 67)
        product = (action - 59).clamp(0, 8)
        amount = torch.minimum(held.gather(1, product[:, None]).squeeze(1), room)
        shed = (
            shed
            + (self.items[None] == product[:, None])
            * torch.where(product_action & at_shed, amount, 0)[:, None]
        )
        plant = (action >= 45) & (action <= 49)
        crop = (action - 45).clamp(0, 4)
        seeds = state.seeds - ((self.crop_ids[None] == crop[:, None]) & plant[:, None]).long()
        ongoing = (species == 2) | (species == 3)
        clear = (action == 53) | ((action == 51) & (kind == 3) & ~ongoing)
        updated = torch.where(clear[:, None], 0, tile)
        zero = torch.zeros_like(kind)
        minus_one = torch.full_like(kind, -1)
        planted = torch.stack(
            (
                torch.full_like(kind, 3),
                crop,
                zero,
                state.day,
                (~((crop == 2) | (crop == 3))).long(),
                zero,
                zero,
                zero,
                minus_one,
            ),
            dim=1,
        )
        updated = torch.where(plant[:, None], planted, updated)
        building = (action == 54) | (action == 55)
        built = torch.stack(
            (action - 50, zero, zero, zero, zero, zero, zero, zero, minus_one), dim=1
        )
        updated = torch.where(building[:, None], built, updated)
        installed = torch.stack(
            (
                structure,
                animal_species,
                torch.ones_like(kind),
                state.day,
                zero,
                zero,
                zero,
                zero,
                minus_one,
            ),
            dim=1,
        )
        updated = torch.where(installs[:, None], installed, updated)
        # Remaining updates do not replace tile identity.
        water = action == 50
        age = state.day - origin
        bonus = torch.where(until >= state.day, 2, 1)
        yield_gain = (
            water
            & ~ongoing
            & (age >= (self.max_yield_day[species] + 1) // 2)
            & (age <= self.max_yield_day[species])
        )
        yield_after = torch.where(
            yield_gain, torch.minimum(stock + bonus, self.max_yield[species]), stock
        )
        yield_after = torch.where(action == 51, 0, yield_after)
        columns = list(updated.unbind(-1))
        columns[4] = torch.where(water | (action == 51), yield_after, columns[4])
        columns[5] = torch.where(water | (action == 56), 1, columns[5])
        columns[6] = torch.where(action == 58, 1, columns[6])
        columns[7] = torch.where(action == 57, 0, columns[7])
        columns[8] = torch.where(action == 52, torch.maximum(until, state.day + 2), columns[8])
        updated = torch.stack(columns, dim=1)
        shared_position = (state.positions == position[:, None]).all(-1) & active[:, None]
        tiles = torch.where(shared_position[..., None], updated[:, None], state.tiles)
        return state._replace(seeds=seeds, shed=shed, tiles=tiles)

    def market_kind_mask(self, state: PolicyLedger) -> Tensor:
        room = state.capacity - state.shed.sum(-1)
        cost = self._hire_cost(state)
        hire = (state.units + state.hires - state.original_hires < 16) & (state.money >= cost)
        land = (state.land < 3) & (state.money >= self.land_cost[state.land])
        goods = torch.stack((state.market[:, 0], state.market[:, 8]), dim=1) - 1
        goods_cost = self.prices[self.items[[0, 8]][None], goods - self.minimum]
        mask = torch.cat(
            (
                torch.ones_like(hire[:, None]),
                hire[:, None],
                land[:, None],
                state.money[:, None] >= self.seed_cost,
                (room[:, None] > 0) & (state.money[:, None] >= goods_cost),
                (room[:, None] > 0) & (state.money[:, None] >= self.animal_cost),
                state.shed[:, :9] > 0,
            ),
            dim=1,
        )
        return mask & (
            state.market_active[:, None] | (torch.arange(22, device=mask.device)[None] == 0)
        )

    def market_quote(self, state: PolicyLedger, kind: Tensor) -> MarketQuote:
        """Cache exact cumulative costs/movements for the subsequent quantity draw."""
        q = self.quantities[None]
        room = (state.capacity - state.shed.sum(-1)).clamp_min(0)
        buying_goods = (kind == 8) | (kind == 9)
        selling = kind >= 13
        item = torch.where(kind == 9, 8, torch.where(selling, kind - 13, 0))
        inventory = state.market.gather(1, item[:, None])
        indices = inventory + torch.where(buying_goods[:, None], -q, q - 1)
        prices = self.prices[item[:, None], indices - self.minimum].long()
        cumulative = _integer_cumsum(prices)
        seed = (kind >= 3) & (kind <= 7)
        animal = (kind >= 10) & (kind <= 12)
        seed_price = self.seed_cost[(kind - 3).clamp(0, 4)]
        animal_price = self.animal_cost[(kind - 10).clamp(0, 2)]
        cost = torch.where(seed[:, None], seed_price[:, None] * q, cumulative)
        cost = torch.where(animal[:, None], animal_price[:, None] * q, cost)
        cost = torch.where((kind == 1)[:, None], self._hire_cost(state)[:, None], cost)
        cost = torch.where((kind == 2)[:, None], self.land_cost[state.land][:, None], cost)
        cost = torch.where((kind == 0)[:, None], 0, cost)
        available = state.shed.gather(1, item[:, None])
        mask = torch.where(selling[:, None], q <= available, cost <= state.money[:, None])
        mask = mask & (~(buying_goods | animal)[:, None] | (q <= room[:, None]))
        mask = torch.where((kind < 3)[:, None] | ~state.market_active[:, None], q == 1, mask)
        market_delta = torch.where(buying_goods[:, None], -q, _integer_cumsum((prices > 1).long()))
        market_delta = torch.where((buying_goods | selling)[:, None], market_delta, 0)
        return MarketQuote(
            kind, mask, torch.where(selling[:, None], cost, -cost), market_delta, item
        )

    def apply_market(
        self, state: PolicyLedger, quote: MarketQuote, quantity: Tensor
    ) -> PolicyLedger:
        """Apply one legal market decision, quantity encoded as the 0..99 bin."""
        kind = quote.kind
        active = state.market_active & (kind != 0)
        quantity = torch.where(kind < 3, 0, quantity)
        amount = quantity + 1
        money_delta = quote.money_delta.gather(1, quantity[:, None]).squeeze(1)
        market_delta = quote.market_delta.gather(1, quantity[:, None]).squeeze(1)
        buying_goods = (kind == 8) | (kind == 9)
        buying_animal = (kind >= 10) & (kind <= 12)
        selling = kind >= 13
        shed_item = torch.where(buying_animal, kind - 1, quote.item)
        stock_delta = torch.where(
            buying_goods | buying_animal, amount, torch.where(selling, -amount, 0)
        )
        shed = (
            state.shed
            + (self.items[None] == shed_item[:, None])
            * torch.where(active, stock_delta, 0)[:, None]
        )
        market = (
            state.market
            + (self.items[:9][None] == quote.item[:, None])
            * torch.where(active, market_delta, 0)[:, None]
        )
        seeds = (
            state.seeds
            + (self.crop_ids[None] == (kind - 3)[:, None])
            * torch.where(active & (kind >= 3) & (kind <= 7), amount, 0)[:, None]
        )
        return state._replace(
            money=state.money + torch.where(active, money_delta, 0),
            hires=state.hires + (active & (kind == 1)),
            land=state.land + (active & (kind == 2)),
            shed=shed,
            market=market,
            seeds=seeds,
            market_active=active,
        )

    def replay_masks(
        self, packed: Tensor, units: Tensor, kinds: Tensor, quantities: Tensor
    ) -> ReplayMasks:
        """Replay supplied factors with the native oracle's invalid-action fallbacks."""
        state = initial_ledger(packed)
        unit_masks, kind_masks, quantity_masks = [], [], []
        market_active, quantity_active = [], []
        for unit in range(16):
            mask = self.unit_mask(state, unit)
            action = units[:, unit]
            selected = torch.where(mask.gather(1, action[:, None]).squeeze(1), action, 0)
            unit_masks.append(mask)
            state = self.apply_unit(state, unit, selected)
        for slot in range(10):
            mask = self.market_kind_mask(state)
            kind = kinds[:, slot]
            selected = torch.where(mask.gather(1, kind[:, None]).squeeze(1), kind, 0)
            quote = self.market_quote(state, selected)
            quantity = quantities[:, slot]
            selected_quantity = torch.where(
                quote.mask.gather(1, quantity[:, None]).squeeze(1), quantity, 0
            )
            kind_masks.append(mask)
            quantity_masks.append(quote.mask)
            market_active.append(state.market_active)
            quantity_active.append(state.market_active & (selected >= 3))
            state = self.apply_market(state, quote, selected_quantity)
        return ReplayMasks(
            torch.stack(unit_masks, 1),
            torch.stack(kind_masks, 1),
            torch.stack(quantity_masks, 1),
            torch.arange(16, device=packed.device)[None] < packed[:, 3, None],
            torch.stack(market_active, 1),
            torch.stack(quantity_active, 1),
        )


def pack_observations(observations: Sequence[dict[str, Any]]) -> np.ndarray:
    """Exact observation adapter for BC preprocessing and public inference.

    Like native collection, this supports the default board/capacity/pricing.
    It rejects custom market curves instead of silently changing affordability.
    """
    packed = np.zeros((len(observations), POLICY_LEDGER_WIDTH), dtype=np.int64)
    packed[:, 257:449] = 12
    tile_kinds = {"LOCKED": 1, "WEED": 2, "PLANT": 3, "COOP": 4, "PASTURE": 5}
    for row, observation in enumerate(observations):
        market = observation.get("market") or {}
        if market.get("params") not in (None, MARKET_PARAMS):
            raise ValueError("device ledger requires native default market parameters")
        farm = observation["farms"][int(observation.get("player", 0))]
        private = observation.get("private") or {}
        positions = [farm["farmer"], *farm.get("hands", [])][:16]
        day = int(observation.get("day", 0))
        packed[row, :7] = (
            day,
            int(farm.get("money", 0)),
            int(farm.get("hires_today", 0)),
            len(positions),
            len(farm.get("unlocked_quadrants", [])) - 1,
            100,
            1,
        )
        packed[row, 7:12] = [int((private.get("seeds") or {}).get(c, 0)) for c in CROPS]
        packed[row, 12:24] = [int((private.get("shed") or {}).get(p, 0)) for p in PRIVATE_ITEMS]
        packed[row, 24:33] = [int((market.get("inventory") or {}).get(p, 10000)) for p in PRODUCTS]
        inventories = private.get("inventories") or []
        tiles = farm["tiles"]
        if len(tiles) != 10 or any(len(line) != 10 for line in tiles):
            raise ValueError("device ledger requires a 10x10 board")
        for unit, (x, y) in enumerate(positions):
            packed[row, 33 + 2 * unit : 35 + 2 * unit] = x, y
            held = inventories[unit] if unit < len(inventories) else {}
            packed[row, 65 + 12 * unit : 77 + 12 * unit] = [
                int(held.get(p, 0)) for p in PRIVATE_ITEMS
            ]
            order = [PRIVATE_ITEMS.index(p) for p, n in held.items() if p in PRIVATE_ITEMS and n]
            packed[row, 257 + 12 * unit : 257 + 12 * unit + len(order)] = order
            tile = tiles[y][x]
            if tile is None:
                continue
            if isinstance(tile, str):
                packed[row, 449 + unit * 9] = tile_kinds[tile]
                continue
            kind = tile["kind"]
            animal = tile.get("animal")
            species = (
                ("GOOSE", "COW", "SHEEP").index(animal)
                if animal
                else (CROPS.index(tile["crop"]) if kind == "PLANT" else 0)
            )
            packed[row, 449 + unit * 9 : 458 + unit * 9] = (
                tile_kinds[kind],
                species,
                bool(animal),
                int(tile.get("placed_day" if animal else "planted_day", 0)),
                int(tile.get("yield_units", 0)),
                bool(tile.get("fed_today" if animal else "watered_today", False)),
                bool(tile.get("cared_today", False)),
                bool(tile.get("fertilizer_available", False)),
                int(
                    tile.get(
                        "fertilized_until_day",
                        -1 if kind in {"WEED", "PLANT", "COOP", "PASTURE"} else 0,
                    )
                ),
            )
    return packed
