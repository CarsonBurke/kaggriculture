# Default Game Constants

These are the defaults in `kaggle-environments==1.32.7`. Configuration overrides
can change values in the first table and the market curve parameters.

## Configuration

| Parameter | Default | Meaning |
|-|-:|-|
| `episodeSteps` | 720 | Recorded states, including the initial state |
| `actTimeout` | 1 s | Base allowance per agent call before overage is charged |
| `boardSize` | 10 | Width and height of each farm |
| `startingMoney` | 3,000 | Starting bank balance per player |
| `turnsPerDay` | 24 | Turns per in-game day |
| `maxMarketOrdersPerTurn` | 10 | Processed order slots per player |
| `shedCapacity` | 100 | Total non-seed items |
| `weedSpawnChance` | 0.005 | Daily chance per empty unlocked tile |
| `farmHandCostMult` | 1 | Multiplier on daily Fibonacci hire costs |
| `townShopUnlockInterval` | 3 days | Shop-unlock cadence |
| `townShopSellInterval` | 4 turns | Shop-demand cadence |
| `townCenterSellInterval` | 24 turns | Town-center demand cadence |
| `seed` | null | Optional deterministic episode seed, hidden from agents after use |
| `marketParams` | `{}` | Sparse per-product price-curve overrides |

Each agent starts with 60 seconds of cumulative framework overage time. Time beyond
`actTimeout` on a call is deducted from that bank rather than resetting the bank
each turn.

## Crops

Ages are whole in-game days since planting.

| Crop | Seed | Type | First yield | Production / bonus ages | Held cap |
|-|-:|-|-:|-|-:|
| Wheat | $10 | One-time | 2 | Water bonus on ages 2-4 | 6 |
| Carrot | $20 | One-time | 2 | Water bonus on ages 2-3 | 4 |
| Tomato | $50 | Ongoing | 8 | Daily at ages 8-11 | 4 |
| Strawberry | $100 | Ongoing | 10 | Ages 10, 12, 14, 16 | 4 |
| Melon | $80 | One-time | 10 | Water bonus on ages 6-12 | 6 |

One-time crops start with one harvestable unit. Ongoing crops have four production
events; their cap applies to unharvested units held on the tile, so regular
harvesting can make lifetime fertilized output exceed the cap.

## Animals

| Animal | Fixed cost | Structure | Product | First yield | Interval | Held cap |
|-|-:|-|-|-:|-:|-:|
| Goose | $300 | Coop | Egg | Age 4 | 1 day | 4 |
| Cow | $400 | Pasture | Milk | Age 8 | 2 days | 6 |
| Sheep | $500 | Pasture | Wool | Age 6 | 3 days | 6 |

## Land and Labor

- Quadrants unlock in fixed order: `NE`, `SW`, then `SE`.
- Their respective prices are $1,000, $2,000, and $4,000.
- The northwest quadrant is free and initially unlocked.
- Successive hands hired in one day cost
  `farmHandCostMult * (1, 1, 2, 3, 5, 8, 13, ...)`.
- The hire count and price sequence reset every day.

## Fixed Limits

- Maximum unlocked town-shop instances: 8.
- Market price floor: $1.
- Initial market inventory per product: 10,000 under the default curves.
- Structures cost no money; building one consumes only that unit's action.
