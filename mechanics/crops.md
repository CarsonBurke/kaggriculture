# Crops

Crops are planted from the player's separate seed inventory onto empty, unlocked
tiles. All crop ages are measured in whole days: `current day - planted day`.

| Crop | Seed | Type | Harvest starts | Water-bonus / production ages | Cap |
|-|-:|-|-:|-|-:|
| Wheat | $10 | One-time | 2 | Bonus on ages 2-4 | 6 |
| Carrot | $20 | One-time | 2 | Bonus on ages 2-3 | 4 |
| Tomato | $50 | Ongoing | 8 | Produces at ages 8, 9, 10, 11 | 4 held |
| Strawberry | $100 | Ongoing | 10 | Produces at ages 10, 12, 14, 16 | 4 held |
| Melon | $80 | One-time | 10 | Bonus on ages 6-12 | 6 |

## Water and Survival

`WATER` marks a plant for the current day. Further water actions that day do
nothing. The flag resets at end of day.

A new plant starts with one missed watering already recorded: the planting day
counts. If it is not watered before that day's refresh, it immediately reaches two
consecutive misses and becomes a weed. After a watered day, a plant may survive one
complete missed day, but a second consecutive missed refresh turns it into a weed.

Watering affects survival for every crop. For ongoing crops it does not affect the
base production unit, but it is required to receive a fertilizer bonus.

## Fertilizer

`FERTILIZE` consumes one fertilizer from the active unit's carried inventory. It
is active on the current day and the next two days. Fertilizing again can extend,
but never shorten, that end day.

The bonus is evaluated when watering or production occurs:

- For a one-time crop, a water action inside its bonus window adds two units instead
  of one while fertilizer is active.
- For an ongoing crop, a scheduled production adds two units instead of one only
  if the plant was both watered and fertilized that day.

Fertilizing after watering does not retroactively increase that day's one-time-crop
bonus.

## One-Time Crops

Wheat, carrot, and melon start with one harvestable unit. A water action within the
crop's bonus window adds one unit, or two with active fertilizer, up to the crop's
cap.

Harvest remains unavailable until the crop's first-yield age. Once available,
`HARVEST` transfers all `yield_units` to that unit's carried inventory and
leaves the tile empty.

With daily watering and no fertilizer:

- Wheat reaches four units at age 4.
- Carrot reaches three units at age 3.
- Melon reaches its six-unit cap at age 10, although its formal bonus window runs
  through age 12.

## Ongoing Crops

Tomato produces once per day at ages 8 through 11. Strawberry produces every other
day at ages 10, 12, 14, and 16. Production occurs during the end-of-day refresh
that advances into the listed age.

Each scheduled event adds one held unit, or two when watered and fertilized, up to
four unharvested units on the tile. `HARVEST` transfers all currently held units
but leaves the plant in place. Because the four-unit limit is a held-inventory cap,
harvesting between fertilized production events can yield more than four units over
the crop's lifetime.

## Decay

An unharvested crop eventually decays:

- A one-time crop starts decaying at the first step of the day after its
  `max_yield_day`.
- An ongoing crop starts decaying one day after its fourth scheduled production.

At the decay boundary and every second turn thereafter, held yield falls by one.
When it reaches zero, the plant becomes a weed. Unit actions resolve before decay,
so a crop can still be harvested on the boundary turn.
