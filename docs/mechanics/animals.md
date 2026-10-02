# Animals

Animals produce indefinitely while they remain on the farm. They require a matching
structure, regular wheat feed, and regular harvesting to avoid their held-product
cap. They may miss one feeding refresh, but not two consecutively.

| Animal | Cost | Home | Product | First production | Interval | Held cap |
|-|-:|-|-|-:|-:|-:|
| Goose | $300 | Coop | Egg | Age 4 | 1 day | 4 |
| Cow | $400 | Pasture | Milk | Age 8 | 2 days | 6 |
| Sheep | $500 | Pasture | Wool | Age 6 | 3 days | 6 |

## Buying and Placement

Animal purchases use `BUY_ANIMAL animal n` at the fixed table price. A successful
purchase goes into the shed and requires free shed capacity.

To place one:

1. Build a free coop or pasture on an empty, unlocked tile.
2. `PICKUP` the animal from the shed.
3. Stand on the matching empty structure and use `PLACE animal`.

Coops accept geese; pastures accept cows and sheep. Placing always consumes one
animal even if a quantity argument is supplied. An occupied structure cannot be
dug up, and there is no action to recover or sell a placed animal.

## Feeding and Escape

`FEED` consumes one carried wheat and marks the animal fed for that day. Repeated
feed actions on the same day do nothing.

A newly placed animal starts with zero missed days, so it survives its placement
day without feed. One missed end-of-day refresh changes the counter to one; a
second consecutive miss makes the animal escape before that day's production.
Its empty coop or pasture remains.

Feeding primarily controls survival and care bonuses. A surviving animal still
produces its base unit on a scheduled day after one missed feeding day.

## Production and Harvest

Age is measured in whole days since placement. Production occurs in the end-of-day
refresh that advances into a scheduled age:

- Geese produce daily from age 4 onward.
- Cows produce every two days from age 8 onward.
- Sheep produce every three days from age 6 onward.

A normal event adds one product. Stored `yield_units` cannot exceed the animal's
held cap; excess production is lost. `HARVEST` transfers all held eggs, milk, or
wool into the active unit's carried inventory and leaves the animal in place.

## Care

`CARE` marks an animal once for the current day and consumes no item. At end of
day, care and production resolve as follows:

1. On a scheduled production day, a fed animal adds its entire previously banked
   care bonus to the base unit, subject to the held cap, then clears that bank.
2. An unfed but surviving animal produces the base unit and loses any banked bonus.
3. After production, a day that was both fed and cared banks one bonus unit for the
   next production event.

Care on a production day therefore applies to a later event, not the event that
just occurred. Caring without feeding banks nothing.

## Fertilizer

Every animal that survives an end-of-day refresh makes one fertilizer available,
whether or not it was fed, cared for, or scheduled to produce. Availability is a
boolean and does not accumulate.

`COLLECT_FERTILIZER` moves that one unit into the active farmer's inventory and
clears the flag. It becomes available again after the next surviving daily refresh.
