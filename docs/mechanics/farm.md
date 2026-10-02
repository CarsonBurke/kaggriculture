# Farm, Land, and Shed

Each player has an independent square grid. Coordinates are `[x, y]`: `x`
increases east, `y` increases south, and `tiles` is indexed as
`tiles[y][x]`.

## Land

The default 10x10 farm is divided into four 5x5 quadrants:

| Quadrant | Initial state | Unlock cost |
|-|-|-:|
| Northwest (`NW`) | Unlocked | Free |
| Northeast (`NE`) | Locked | $1,000 |
| Southwest (`SW`) | Locked | $2,000 |
| Southeast (`SE`) | Locked | $4,000 |

`BUY_LAND` always buys the next quadrant in that order; the player cannot choose
a different one. A locked tile is represented by `"LOCKED"`.

Farmers may stand on and walk across locked tiles. Normal tile actions, including
planting, building, watering, harvesting, feeding, and digging, do nothing there.
Shed operations are the exception.

Multiple farmers and hands may occupy the same coordinate. Their positions do not
occupy the underlying farm tile and do not block movement or tile construction.

## Central Shed

The shed is not present in `tiles`. It is accessible from the four inner-corner
tiles around the center:

```text
(half - 1, half - 1)  (half, half - 1)
(half - 1, half)      (half, half)
```

Here `half = boardSize // 2`. On a 10x10 board the access tiles are `(4, 4)`,
`(5, 4)`, `(4, 5)`, and `(5, 5)`. Shed actions work from all four even when
the standing tile is locked.

The private inventory has three layers:

- **Seeds:** Separate, unlimited storage. `PLANT` consumes them directly.
- **Shed:** Animals, fertilizer, and products, with a shared capacity of 100 by
  default.
- **Carried inventories:** One unbounded dictionary per farmer or current hand.

Only items in the shed may be sold. Feed and fertilizer actions consume carried
items, so a unit normally has to pick them up first.

## Shed Actions

- `PICKUP item [n]` moves up to `n` items from the shed to the active unit.
  The default is one. Seeds cannot be picked up.
- `PLACE item [n]` at a shed-access tile moves up to `n` carried items into
  available shed space. Items that do not fit remain carried.
- `DROP` at a shed-access tile empties the unit's entire inventory. Items fill
  remaining shed space and all overflow is discarded.
- At end of day, every carried inventory is dropped automatically. Overflow is
  likewise discarded.

When space is scarce, explicit `DROP` follows that inventory dictionary's insertion
order. The automatic drop processes the main farmer and then hands in unit order,
using each inventory's insertion order. Deposit valuable items explicitly before
the refresh rather than relying on overflow order.

`BUY_PRODUCT` and `BUY_ANIMAL` deposit directly into the shed and stop without
charging when it is full. Seeds do not count against the capacity.

## Structures

- `BUILD_COOP` creates an empty coop on the current empty, unlocked tile.
- `BUILD_PASTURE` creates an empty pasture on the current empty, unlocked tile.
- Building a structure costs no money.
- `DIG` clears a plant, weed, or empty structure without returning anything.
- An occupied coop or pasture cannot be dug up. If its animal escapes, the empty
  structure remains and can then be removed.

## Weeds

At every end-of-day refresh, each empty unlocked tile independently has the
configured weed-spawn chance, 0.005 by default. Plant neglect and crop decay can
also replace a plant with a weed. A weed blocks planting and construction until a
unit uses `DIG` while standing on it.

Episode randomness is deterministic for a resolved episode seed, including weeds
and town-shop selection.
