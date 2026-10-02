# Game Overview

These notes describe the rules implemented by the repository's pinned
`kaggle-environments==1.32.7`, using the default configuration unless stated
otherwise.

Kaggriculture is a two-player farming simulation. Each player controls a separate
10x10 farm, starts with $3,000, and competes over a nominal 30-day season. A day is
24 turns.

**Objective:** Finish with more money in the bank than the other player. The final
bank balances are also the agents' rewards, so ties are possible. Seeds, crops,
animals, carried goods, and shed contents have no terminal value unless sold.

## Starting State

- Only the northwest 5x5 quadrant of each farm is unlocked.
- The main farmer starts on the northwest tile next to the central shed: `(4, 4)`
  on the default board.
- The shed, seed inventory, and carried inventories are empty.
- Every market product starts at its equilibrium inventory and base price.
- No town shops are unlocked.

The two farms, bank balances, farmer positions, market, and town are public.
Shed contents, seed counts, and carried inventories are private to their owner.

## Turn Structure

Agents choose their complete actions from the same pre-turn state. The environment
then resolves a turn in this order:

1. For each player, apply the main farmer's action and then each listed farm-hand
   action. A player's over-requested `PLANT` actions are rejected atomically per
   crop before any of its units act.
2. Process market-order slots in order. `HIRE` and `BUY_LAND` resolve once;
   product orders resolve one unit at a time with both players quoted before either
   unit is committed.
3. Apply town-center and unlocked-shop demand.
4. Decay over-age plants.
5. At the end of a day, refresh plants and animals, spawn weeds, drop carried
   inventories into sheds, reset farmers, remove hired hands, and possibly unlock a
   town shop.
6. Advance `step`, `day`, and `hour`; on the terminal state, set each reward
   to that player's bank balance.

Farmer actions are applied sequentially within a farm even though they are selected
together. This allows a later hand to interact with a change made by the farmer or
an earlier hand in the same turn. Players cannot act on each other's farms, so the
player-resolution order normally has no cross-player effect.

Well-formed but illegal or unaffordable actions are silent no-ops. Malformed nested
values are not comprehensively validated: for example, a non-integer quantity or
non-string opcode can raise from the interpreter. Agents should always emit the
documented list shapes, string opcodes, and integer quantities.

## Episode-Length Detail

`episodeSteps=720` means 720 recorded environment states, including the initial
state. In this pinned release, a default local run calls each active agent 719
times, for steps 0 through 718, and records the terminal state at step 719. Thus the
last actionable hour is day 29, hour 22; day 29, hour 23 is terminal state only.

## Mechanics Pages

- [Agent API and observations](api.md)
- [Default constants](constants.md)
- [Farm, land, shed, and weeds](farm.md)
- [Farmers and farm hands](farmers.md)
- [Crop growth and harvest](crops.md)
- [Animal production and care](animals.md)
- [Market and dynamic prices](market.md)
- [Town demand](town.md)
