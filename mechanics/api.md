# Agent API

An agent is a Python function that receives an observation and returns one action
dictionary. It may optionally accept the resolved environment configuration:

```python
def agent(obs):
    return {
        "farmer": ["PASS"],
        "hands": [],
        "market": [],
    }


def agent_with_configuration(obs, config):
    return {"farmer": ["PASS"], "hands": [], "market": []}
```

Missing top-level fields receive default actions. Well-formed but illegal actions,
unknown string operations, unavailable items, and unaffordable orders fail
silently. Nested action contents are not comprehensively schema-validated: malformed
quantities or non-string opcodes can raise and abort a local run. Always use the
documented list shapes, string opcodes, and integer quantities.

## Observation

The top-level fields are:

- `player`: this agent's player index, 0 or 1.
- `step`: zero-indexed framework step.
- `day`: `step // turnsPerDay`.
- `hour`: `step % turnsPerDay`.
- `farms`: public farm state for every player.
- `market`: shared product inventories and current prices.
- `town`: shared list of unlocked shop instances.
- `private`: this player's shed, seeds, and unit inventories.
- `remainingOverageTime`: cumulative framework-managed overage allowance remaining
  for this agent.

Each public farm has this shape:

```python
{
    "money": float,
    "tiles": [[tile, ...], ...],  # indexed tiles[y][x]
    "farmer": [x, y],
    "hands": [[x, y], ...],
    "unlocked_quadrants": ["NW", ...],
    "hires_today": int,
}
```

The private state is:

```python
{
    "shed": {item: count, ...},
    "seeds": {crop: count, ...},
    "inventories": [
        {item: count, ...},            # main farmer
        {item: count, ...},            # first hand, then remaining hands
    ],
}
```

## Tile Values

A tile is one of:

- `None`: empty, unlocked land.
- `"LOCKED"`: land not yet purchased.
- `{"kind": "WEED"}`.
- A plant dictionary:

```python
{
    "kind": "PLANT",
    "crop": "WHEAT",  # or another crop
    "planted_day": int,
    "watered_today": bool,
    "consecutive_unwatered": int,
    "yield_units": int,
    "max_lifespan_step": int,
    "fertilized_until_day": int,
}
```

- An empty structure, `{"kind": "COOP"}` or `{"kind": "PASTURE"}`.
- An occupied animal structure:

```python
{
    "kind": "COOP",  # or PASTURE
    "animal": "GOOSE",  # or COW / SHEEP
    "placed_day": int,
    "yield_units": int,
    "fed_today": bool,
    "consecutive_unfed": int,
    "cared_today": bool,
    "fertilizer_available": bool,
    "pending_care_bonus": int,
}
```

## Farmer and Hand Actions

`farmer` contains one operation. `hands` contains one operation for each current
hand, in the same order as `farms[player]["hands"]`. Missing entries pass. Extra
entries are not executed, but extra `PLANT` requests still count during atomic
seed validation and can block valid units' planting.

- Movement: `["NORTH"]`, `["SOUTH"]`, `["EAST"]`, `["WEST"]`, `["PASS"]`
- Shed: `["PICKUP", item, n?]`, `["PLACE", item, n?]`, `["DROP"]`
- Crops: `["PLANT", crop]`, `["WATER"]`, `["HARVEST"]`, `["FERTILIZE"]`
- Animals: `["BUILD_COOP"]`, `["BUILD_PASTURE"]`, `["PLACE", animal]`,
  `["FEED"]`, `["HARVEST"]`, `["CARE"]`, `["COLLECT_FERTILIZER"]`
- Terrain: `["DIG"]`

All non-movement tile actions target the tile under that unit. Each unit may execute
at most one action per turn.

## Market Actions

`market` is an ordered list, capped at `maxMarketOrdersPerTurn`:

- `["BUY_SEED", crop, n]`
- `["BUY_PRODUCT", item, n]` for wheat or fertilizer only
- `["BUY_ANIMAL", animal, n]`
- `["SELL", product, n]`
- `["HIRE"]`
- `["BUY_LAND"]`

Quantified orders stop early if money, the seller's shed stock, or destination shed
capacity runs out. Seeds and animals have fixed purchase prices; products use the
dynamic market price.

## Configuration

An agent with a second positional argument receives the resolved configuration,
including board and timing settings. The optional episode seed is consumed during
initialization and cleared from the agent-visible configuration; agents cannot use
`config.seed` to forecast weeds or shop draws.

`actTimeout` is the base allowance for each agent call. Only time beyond that
allowance is deducted from the agent's cumulative `remainingOverageTime`, which
starts at 60 seconds by default. Once a call's excess time is greater than the
remaining bank, the framework replaces its action with a timeout.

## Built-in Agents

The environment accepts `"pass"`, `"random"`, and `"starter"`. The starter is
a deterministic carrot-growing baseline.
