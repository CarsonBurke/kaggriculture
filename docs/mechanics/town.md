# Town Demand

The town removes products from shared market inventory. It pays neither player;
its strategic effect is to create scarcity and raise market prices.

## Town Center

At every `townCenterSellInterval` step, 24 turns by default, the town center
removes one of every product except fertilizer:

`WHEAT`, `CARROT`, `TOMATO`, `STRAWBERRY`, `MELON`, `EGG`, `MILK`,
and `WOOL`.

The first tick is turn step 0, after player market orders. With default settings it
then ticks once at the start of each in-game day.

## Shops

One shop instance unlocks whenever the new day number is divisible by
`townShopUnlockInterval`, every three days by default. The first unlock occurs in
the end-of-day transition into day 3.

Selection is uniform with replacement, so duplicate shop names are meaningful.
Each instance remains active and consumes independently. Unlocking stops after
eight total instances.

| Shop identifier | Products consumed per tick |
|-|-|
| `BAKERY` | Egg, wheat |
| `PIZZA_SHOP` | Milk, tomato, wheat |
| `BRUNCH_SPOT` | Egg, wheat, strawberry |
| `YARN_STORE` | Wool x2 |
| `ICE_CREAM_SHOP` | Strawberry, milk, wheat |
| `PET_CAFE` | Carrot x2 |
| `SMOOTHIE_SHOP` | Strawberry, milk |
| `FARMERS_MARKET` | Wheat, carrot, tomato, strawberry |

Every active shop ticks at `townShopSellInterval`, every four turns by default.
A multi-product shop removes one of each listed product per tick. A single-product
shop removes two units.

For example, two unlocked yarn-store instances remove four wool every shop tick.
The town can unlock several copies of one shop and no copies of another.

## Timing and Prices

Player market orders resolve before town demand. Shop demand and town-center demand
then apply on the same turn if both intervals divide the current step. Prices are
refreshed after all town consumption, so the next observation exposes the new
inventory and price.

The episode seed makes the sequence reproducible, but shop selection shares the
day's seeded random stream with weed spawning.
