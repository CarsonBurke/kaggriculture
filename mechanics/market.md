# Market

Seeds and animals have unlimited supply and fixed purchase prices. Harvested
products trade against shared inventory-dependent prices that persist for the
entire episode.

## Valid Trades

| Order | Source / destination | Price |
|-|-|-|
| `BUY_SEED crop n` | Unlimited supply to seed inventory | Fixed seed cost |
| `BUY_ANIMAL animal n` | Unlimited supply to shed | Fixed animal cost |
| `BUY_PRODUCT WHEAT n` | Market inventory to shed | Dynamic |
| `BUY_PRODUCT FERTILIZER n` | Market inventory to shed | Dynamic |
| `SELL product n` | Shed to market | Dynamic |

Every crop and animal product, plus fertilizer, may be sold. Other products cannot
be bought back. Animals and seeds cannot be sold.

Buying an animal or product requires shed space. Seeds bypass the shed. Selling
draws only from the shed, not a unit's carried inventory.

## Order Resolution

An agent may submit ten order slots per turn by default; later entries are dropped.
The environment processes slot 0 for both players, then slot 1, and so on.

`HIRE` and `BUY_LAND` are atomic orders and resolve once in player order. A
quantified trade resolves one unit at a time:

1. Quote each player's next unit against the same pre-commit market inventory.
2. Commit player 0's and then player 1's quoted unit.
3. Repeat until each quantity completes or can no longer proceed.

This prevents the first player from receiving a different quote merely because its
same-slot unit commits first. A large order finishes before processing the next
order slot.

An order stops at its first insufficient-money, insufficient seller-stock, or
full-destination-shed failure. Market inventory itself is not clamped or checked
before a product buy; the default inventory is deliberately much larger than
realistic demand. Invalid orders are silent no-ops.

## Inventory Effects

- Selling quotes the current, pre-sale inventory, pays that amount, and normally
  adds one unit to market inventory.
- Buying quotes the price at inventory minus one, charges it, and subtracts one
  unit from market inventory.
- This quote convention makes an immediate buy followed by a sell against an
  otherwise unchanged market break even.
- A sale made at the $1 price floor is accepted but does **not** add market
  inventory.
- Town demand and product purchases reduce inventory; player sales increase it.

## Price Function

Every default product begins at equilibrium inventory `I0 = 10,000`, where its
price equals `base`. Scarcity raises price and a glut lowers it:

```text
distance = abs(inventory - I0)
amplitude = target * base / shape(T)

if inventory < I0:
    price = base + amplitude_below * shape_below(distance)
else:
    price = base - amplitude_above * shape_above(distance)

quoted price = max(1, round(price))
```

`T` is a calibration throughput. `target` says how much of the base price a
movement of `T` units changes on that side. Available shapes are:

- `linear(x) = x`
- `sq(x) = x * x`
- `sqrt(x) = square root of x`
- `log(x) = natural log of (1 + x)`
- `log10(x) = base-10 log of (1 + x)`
- `hinge(x) = u + 8 * max(0, u - 1)^2` with `u = x / T`: linear in `u` up to
  the knee at `T`, then quadratic, so the price holds near base until demand
  outruns a field's output and then runs away. It is the one shape scaled by
  `T`, so `hinge(T) = 1` and `target` keeps its meaning.

## Default Curves

“Below” means scarce inventory below `I0`; “above” means excess inventory above
`I0`.

| Product | Base | T | Below shape / target | Above shape / target |
|-|-:|-:|-|-|
| Wheat | $25 | 400 | sqrt / 0.80 | log / 0.20 |
| Carrot | $35 | 450 | hinge / 1.00 | sqrt / 0.70 |
| Tomato | $60 | 200 | hinge / 0.40 | sqrt / 0.60 |
| Strawberry | $120 | 100 | sqrt / 0.70 | linear / 1.60 |
| Melon | $250 | 300 | log / 0.20 | sq / 3.60 |
| Egg | $50 | 332 | hinge / 0.40 | log / 0.20 |
| Milk | $160 | 122 | sqrt / 0.60 | linear / 1.60 |
| Wool | $200 | 105 | log / 0.20 | sq / 3.20 |
| Fertilizer | $100 | 200 | linear / 0.40 | linear / 0.40 |

`marketParams` may override any subset of `base`, `I0`, `T`,
`below_func`, `below_target`, `above_func`, or `above_target` separately for
each product.
