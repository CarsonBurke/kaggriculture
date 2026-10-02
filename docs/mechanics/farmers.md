# Farmers and Farm Hands

Every player has one permanent main farmer. Hired farm hands provide additional
unit actions for the rest of the current day.

## Movement and Actions

Each unit receives at most one action per turn. Movement is one tile north, south,
east, or west; diagonal movement is unavailable. Moving beyond the board is a
no-op.

Units:

- may cross locked land;
- may share coordinates with any number of friendly units;
- carry independent, unbounded inventories;
- act on the tile under their current position; and
- are ordered as the main farmer followed by hands in spawn order.

Actions are chosen together but applied in that unit order. For example, a farmer
can build a coop and a later hand on the same coordinate can place a goose in it
during the same turn.

## Hiring

`HIRE` is a market order. The daily prices follow the Fibonacci sequence:

```text
farmHandCostMult * (1, 1, 2, 3, 5, 8, 13, 21, ...)
```

With the default multiplier, the first two hands each cost $1. The counter increases
only for successful hires and resets at the end of the day. A hire fails if the
player cannot pay.

A new hand appears on the least-occupied shed-access tile. Ties prefer northwest,
northeast, southwest, then southeast. Locked status is ignored, so the first hand
normally appears at `(5, 4)`, in the initially locked northeast quadrant.

Hiring is resolved after unit actions. A hand therefore cannot act on its hiring
turn, and a hand hired on the last turn of a day disappears immediately in that
turn's end-of-day reset.

## Daily Reset

At the end of each day:

1. All carried inventories are deposited into the shed up to its capacity; overflow
   is lost. The main farmer deposits first, followed by hands in unit order; items
   within an inventory follow dictionary insertion order.
2. The main farmer returns to the northwest shed-access tile.
3. Every hired hand disappears.
4. The hand list and daily hire counter reset.

Hands must be hired again each day. The main farmer persists, but its carried
inventory does not.

## Plant-Request Atomicity

Before a player's units act, the environment counts every `PLANT crop` request in
that player's submitted farmer and hand actions. If the number requested for a crop
exceeds the available seeds, **all** plant requests for that crop become passes for
the turn. This check happens before location legality is tested, so even a request
made from an occupied or locked tile contributes to the count.
