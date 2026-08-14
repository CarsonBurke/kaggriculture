use crate::rng::PyRandom;
use serde::Serialize;

pub const PLAYERS: usize = 2;
pub const BOARD_SIZE: usize = 10;
pub const TILE_COUNT: usize = BOARD_SIZE * BOARD_SIZE;
pub const PRODUCTS: usize = 9;
pub const CROPS: usize = 5;
pub const ANIMALS: usize = 3;
pub const PRIVATE_ITEMS: usize = 12;
pub const MAX_UNITS: usize = 16;
pub const MAX_MARKET_ORDERS: usize = 10;
pub const UNIT_ACTIONS: usize = 59;
pub const MARKET_KINDS: usize = 22;
pub const MARKET_QUANTITIES: usize = 100;
const UNIT_MASK_VALUES: usize = MAX_UNITS * UNIT_ACTIONS;
const MARKET_KIND_MASK_VALUES: usize = MAX_MARKET_ORDERS * MARKET_KINDS;
const MARKET_QUANTITY_MASK_VALUES: usize = MAX_MARKET_ORDERS * MARKET_QUANTITIES;
pub const MAX_SAFE_SEED: u64 = u64::MAX / 1_000_003;
pub const FARM_CHANNELS: usize = 29;
pub const BOARD_CHANNELS: usize = FARM_CHANNELS * 2;
pub const GLOBAL_FEATURES: usize = 72;
pub const CRITIC_FEATURES: usize = 101;
pub const UNIT_FEATURES: usize = 17;
// Structured token widths. Must stay identical to src/kaggriculture/tokens.py.
pub const TILE_TOKENS: usize = TILE_COUNT * PLAYERS;
pub const TILE_CATEGORICAL: usize = 6;
pub const TILE_CONTINUOUS: usize = 18;
pub const UNIT_CATEGORICAL: usize = 4;
pub const UNIT_CONTINUOUS: usize = PRIVATE_ITEMS + 2;
pub const UNIT_GATHERS: usize = 5;
pub const PRODUCT_TOKEN_FIELDS: usize = 5;
pub const CROP_TOKEN_FIELDS: usize = 6;
pub const FARM_TOKEN_FIELDS: usize = 4;
pub const TOWN_TOKEN_FIELDS: usize = 14;

const PRODUCT_NAMES: [&str; PRODUCTS] = [
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
];
const CROP_NAMES: [&str; CROPS] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"];
const ANIMAL_NAMES: [&str; ANIMALS] = ["GOOSE", "COW", "SHEEP"];
const PRIVATE_NAMES: [&str; PRIVATE_ITEMS] = [
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
    "GOOSE",
    "COW",
    "SHEEP",
];
const SHOP_NAMES_SORTED: [&str; 8] = [
    "BAKERY",
    "BRUNCH_SPOT",
    "FARMERS_MARKET",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "PIZZA_SHOP",
    "SMOOTHIE_SHOP",
    "YARN_STORE",
];
const SHOP_PRODUCTS: [&[usize]; 8] = [
    &[5, 0],
    &[5, 0, 3],
    &[0, 1, 2, 3],
    &[3, 6, 0],
    &[1],
    &[6, 2, 0],
    &[3, 6],
    &[7],
];
const SEED_COST: [i64; CROPS] = [10, 20, 50, 100, 80];
const FIRST_YIELD: [u16; CROPS] = [2, 2, 8, 10, 10];
const MAX_YIELD_DAY: [u16; CROPS] = [4, 3, 8, 10, 12];
const CROP_INTERVAL: [u16; CROPS] = [0, 0, 1, 2, 0];
const CROP_MAX_HELD: [u8; CROPS] = [6, 4, 4, 4, 6];
const CROP_ONGOING: [bool; CROPS] = [false, false, true, true, false];
const ANIMAL_COST: [i64; ANIMALS] = [300, 400, 500];
const ANIMAL_FIRST_YIELD: [u16; ANIMALS] = [4, 8, 6];
const ANIMAL_INTERVAL: [u16; ANIMALS] = [1, 2, 3];
const ANIMAL_MAX_HELD: [u8; ANIMALS] = [4, 6, 6];
const ANIMAL_PRODUCT: [usize; ANIMALS] = [5, 6, 7];
const LAND_PRICES: [i64; 3] = [1000, 2000, 4000];
const PRICE_FLOOR: i64 = 1;
const MARKET_I0: i32 = 10_000;

// Illiquid cost-basis credit fractions for the shaping potential.  Kept near
// engine cost so buying an asset is only a small potential dip: a deep dip
// (land once sat at 0.45) makes every purchase an immediate shaped-reward
// cliff the policy never crosses, starving the critic of post-purchase data.
// Must stay identical to ILLIQUID_* in src/kaggriculture/encoding.py.
const ILLIQUID_SHED_ANIMAL_CREDIT: f64 = 0.82;
const ILLIQUID_SHED_SEED_CREDIT: f64 = 0.85;
const ILLIQUID_PLACED_ANIMAL_CREDIT: f64 = 0.85;
const ILLIQUID_PLANTED_SEED_CREDIT: f64 = 0.8;
const ILLIQUID_PENDING_YIELD_CREDIT: f64 = 0.72;
const ILLIQUID_LAND_CREDIT: f64 = 0.9;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum TileKind {
    #[default]
    Empty = 0,
    Locked = 1,
    Weed = 2,
    Plant = 3,
    Coop = 4,
    Pasture = 5,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tile {
    pub kind: TileKind,
    /// 0..4 crop, or 0..2 animal. Meaning is selected by `kind` and `has_animal`.
    pub species: u8,
    pub has_animal: bool,
    pub origin_day: u16,
    pub yield_units: u8,
    pub consecutive_unmet: u8,
    pub watered_or_fed: bool,
    pub cared_today: bool,
    pub fertilizer_available: bool,
    pub pending_care_bonus: u8,
    pub max_lifespan_step: i16,
    pub fertilized_until_day: i16,
}

impl Tile {
    fn plant(crop: usize, day: u16, turns_per_day: u16) -> Self {
        let ongoing = CROP_ONGOING[crop];
        Self {
            kind: TileKind::Plant,
            species: crop as u8,
            origin_day: day,
            yield_units: if ongoing { 0 } else { 1 },
            consecutive_unmet: 1,
            max_lifespan_step: if ongoing {
                -1
            } else {
                ((day + MAX_YIELD_DAY[crop] + 1) * turns_per_day) as i16
            },
            fertilized_until_day: -1,
            ..Self::default()
        }
    }

    fn structure(kind: TileKind) -> Self {
        Self {
            kind,
            max_lifespan_step: -1,
            fertilized_until_day: -1,
            ..Self::default()
        }
    }

    fn animal(animal: usize, day: u16) -> Self {
        Self {
            kind: animal_structure(animal),
            species: animal as u8,
            has_animal: true,
            origin_day: day,
            max_lifespan_step: -1,
            fertilized_until_day: -1,
            ..Self::default()
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Position(pub u8, pub u8);

#[derive(Clone, Debug)]
pub struct Farm {
    pub money: i64,
    pub tiles: [Tile; TILE_COUNT],
    pub positions: [Position; MAX_UNITS],
    pub units: u8,
    /// Bit 0 NW, bit 1 NE, bit 2 SW, bit 3 SE.
    pub unlocked: u8,
    pub hires_today: u8,
}

#[derive(Clone, Debug)]
pub struct PrivateState {
    pub shed: [u16; PRIVATE_ITEMS],
    pub seeds: [u16; CROPS],
    pub inventories: [[u16; PRIVATE_ITEMS]; MAX_UNITS],
    /// Python dict insertion order for carried items. `u8::MAX` is unused.
    pub inventory_order: [[u8; PRIVATE_ITEMS]; MAX_UNITS],
}

#[derive(Clone, Debug)]
pub struct GameConfig {
    pub episode_steps: u16,
    pub turns_per_day: u16,
    pub starting_money: i64,
    pub shed_capacity: u16,
    pub weed_spawn_chance: f64,
    pub shop_unlock_interval: u16,
    pub shop_sell_interval: u16,
    pub town_center_sell_interval: u16,
    pub farm_hand_cost_mult: i64,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            episode_steps: 720,
            turns_per_day: 24,
            starting_money: 3000,
            shed_capacity: 100,
            weed_spawn_chance: 0.005,
            shop_unlock_interval: 3,
            shop_sell_interval: 4,
            town_center_sell_interval: 24,
            farm_hand_cost_mult: 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Game {
    pub seed: u64,
    pub config: GameConfig,
    /// Number of already-applied joint actions. Initial observation is step 0.
    pub step: u16,
    pub done: bool,
    pub farms: [Farm; PLAYERS],
    pub privates: [PrivateState; PLAYERS],
    pub market_inventory: [i32; PRODUCTS],
    pub market_prices: [i64; PRODUCTS],
    /// Shop indices in unlock order; duplicates are meaningful.
    pub shops: [u8; 8],
    pub shop_count: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct CompactAction {
    pub units: [u8; MAX_UNITS],
    pub market_kinds: [u8; MAX_MARKET_ORDERS],
    /// Exact quantity index: 0..99 decodes to 1..100.
    pub market_quantities: [u8; MAX_MARKET_ORDERS],
}

impl Default for CompactAction {
    fn default() -> Self {
        Self {
            units: [0; MAX_UNITS],
            market_kinds: [0; MAX_MARKET_ORDERS],
            market_quantities: [0; MAX_MARKET_ORDERS],
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct MarketOrder {
    kind: u8,
    item: usize,
    remaining: u16,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StepResult {
    pub done: bool,
    pub rewards: [f32; PLAYERS],
    pub money: [f32; PLAYERS],
}

pub struct FactorMasks {
    pub unit: [bool; UNIT_MASK_VALUES],
    pub market_kind: [bool; MARKET_KIND_MASK_VALUES],
    pub market_quantity: [bool; MARKET_QUANTITY_MASK_VALUES],
    pub unit_active: [bool; MAX_UNITS],
    pub market_active: [bool; MAX_MARKET_ORDERS],
    pub market_quantity_active: [bool; MAX_MARKET_ORDERS],
}

impl Default for FactorMasks {
    fn default() -> Self {
        Self {
            unit: [false; UNIT_MASK_VALUES],
            market_kind: [false; MARKET_KIND_MASK_VALUES],
            market_quantity: [false; MARKET_QUANTITY_MASK_VALUES],
            unit_active: [false; MAX_UNITS],
            market_active: [false; MAX_MARKET_ORDERS],
            market_quantity_active: [false; MAX_MARKET_ORDERS],
        }
    }
}

pub struct SampledFactors {
    pub action: CompactAction,
    pub masks: FactorMasks,
    pub unit_logprobs: [f32; MAX_UNITS],
    pub market_kind_logprobs: [f32; MAX_MARKET_ORDERS],
    pub market_quantity_logprobs: [f32; MAX_MARKET_ORDERS],
    pub unit_entropies: [f32; MAX_UNITS],
    pub market_kind_entropies: [f32; MAX_MARKET_ORDERS],
    pub market_quantity_entropies: [f32; MAX_MARKET_ORDERS],
    pub mean_entropy: f32,
}

impl Default for SampledFactors {
    fn default() -> Self {
        Self {
            action: CompactAction::default(),
            masks: FactorMasks::default(),
            unit_logprobs: [0.0; MAX_UNITS],
            market_kind_logprobs: [0.0; MAX_MARKET_ORDERS],
            market_quantity_logprobs: [0.0; MAX_MARKET_ORDERS],
            unit_entropies: [0.0; MAX_UNITS],
            market_kind_entropies: [0.0; MAX_MARKET_ORDERS],
            market_quantity_entropies: [0.0; MAX_MARKET_ORDERS],
            mean_entropy: 0.0,
        }
    }
}

pub struct QuantityHead<'a> {
    pub rank: usize,
    pub kind_gate: &'a [f32],
    pub values: &'a [f32],
    pub bias: &'a [f32],
}

#[derive(Clone)]
struct UnitLedger {
    farm: Farm,
    private: PrivateState,
    config: GameConfig,
}

impl UnitLedger {
    fn from_game(game: &Game, player: usize) -> Self {
        Self {
            farm: game.farms[player].clone(),
            private: game.privates[player].clone(),
            config: game.config.clone(),
        }
    }

    fn action_valid(&self, unit: usize, action: u8, day: u16) -> bool {
        if action >= UNIT_ACTIONS as u8 || unit >= usize::from(self.farm.units) {
            return false;
        }
        if action == 0 {
            return true;
        }
        let position = self.farm.positions[unit];
        let x = usize::from(position.0);
        let y = usize::from(position.1);
        if let Some((dx, dy)) = move_delta(action) {
            let nx = x as i16 + dx;
            let ny = y as i16 + dy;
            return (0..BOARD_SIZE as i16).contains(&nx) && (0..BOARD_SIZE as i16).contains(&ny);
        }
        let at_shed = is_shed_access(x, y);
        if action == 5 {
            return at_shed
                && self.shed_total() < self.config.shed_capacity
                && self.private.inventories[unit]
                    .iter()
                    .any(|&quantity| quantity > 0);
        }
        if let Some((item, quantity)) = pickup_spec(action) {
            return at_shed && self.private.shed[item] >= quantity;
        }
        let tile = self.farm.tiles[y * BOARD_SIZE + x];
        if let Some(animal) = place_animal(action) {
            let has_animal = self.private.inventories[unit][9 + animal] > 0;
            let installs_animal = tile.kind == animal_structure(animal) && !tile.has_animal;
            let deposits_animal = at_shed && self.shed_total() < self.config.shed_capacity;
            return has_animal && (installs_animal || deposits_animal);
        }
        if tile.kind == TileKind::Locked {
            return false;
        }
        if let Some(crop) = unit_plant_crop(action) {
            return tile.kind == TileKind::Empty && self.private.seeds[crop] > 0;
        }
        match action {
            50 => tile.kind == TileKind::Plant && !tile.watered_or_fed,
            51 if tile.kind == TileKind::Plant => {
                tile.yield_units > 0
                    && day.saturating_sub(tile.origin_day) >= FIRST_YIELD[usize::from(tile.species)]
            }
            51 => tile.has_animal && tile.yield_units > 0,
            52 => {
                tile.kind == TileKind::Plant
                    && self.private.inventories[unit][8] > 0
                    && tile.fertilized_until_day < day as i16 + 2
            }
            53 => tile.kind != TileKind::Empty && !tile.has_animal,
            54 | 55 => tile.kind == TileKind::Empty,
            56 => tile.has_animal && !tile.watered_or_fed && self.private.inventories[unit][0] > 0,
            57 => tile.has_animal && tile.fertilizer_available,
            58 => tile.has_animal && !tile.cared_today,
            _ => false,
        }
    }

    fn apply_action(&mut self, unit: usize, action: u8, day: u16) {
        if action >= UNIT_ACTIONS as u8 || unit >= usize::from(self.farm.units) {
            return;
        }
        let position = self.farm.positions[unit];
        let x = usize::from(position.0);
        let y = usize::from(position.1);
        if let Some((dx, dy)) = move_delta(action) {
            self.farm.positions[unit] = Position((x as i16 + dx) as u8, (y as i16 + dy) as u8);
            return;
        }
        if action == 0 {
            return;
        }
        if action == 5 {
            self.drop_inventory(unit);
            return;
        }
        if let Some((item, requested)) = pickup_spec(action) {
            let quantity = self.private.shed[item].min(requested);
            self.private.shed[item] -= quantity;
            self.add_inventory(unit, item, quantity);
            return;
        }
        let tile_index = y * BOARD_SIZE + x;
        let tile = self.farm.tiles[tile_index];
        if let Some(animal) = place_animal(action) {
            let item = 9 + animal;
            if tile.kind == animal_structure(animal) && !tile.has_animal {
                self.take_inventory(unit, item, 1);
                self.farm.tiles[tile_index] = Tile::animal(animal, day);
            } else if is_shed_access(x, y) {
                self.place_to_shed(unit, item, 1);
            }
            return;
        }
        if let Some(crop) = unit_plant_crop(action) {
            self.private.seeds[crop] -= 1;
            self.farm.tiles[tile_index] = Tile::plant(crop, day, self.config.turns_per_day);
            return;
        }
        match action {
            50 => {
                let crop = usize::from(tile.species);
                let mutable = &mut self.farm.tiles[tile_index];
                mutable.watered_or_fed = true;
                if !CROP_ONGOING[crop] {
                    let age = day - tile.origin_day;
                    let start = MAX_YIELD_DAY[crop].div_ceil(2);
                    if (start..=MAX_YIELD_DAY[crop]).contains(&age) {
                        let bonus = if tile.fertilized_until_day >= day as i16 {
                            2
                        } else {
                            1
                        };
                        mutable.yield_units =
                            CROP_MAX_HELD[crop].min(mutable.yield_units.saturating_add(bonus));
                    }
                }
            }
            51 => {
                if tile.kind == TileKind::Plant {
                    let crop = usize::from(tile.species);
                    self.add_inventory(unit, crop, u16::from(tile.yield_units));
                    if CROP_ONGOING[crop] {
                        self.farm.tiles[tile_index].yield_units = 0;
                    } else {
                        self.farm.tiles[tile_index] = Tile::default();
                    }
                } else {
                    let product = ANIMAL_PRODUCT[usize::from(tile.species)];
                    self.add_inventory(unit, product, u16::from(tile.yield_units));
                    self.farm.tiles[tile_index].yield_units = 0;
                }
            }
            52 => {
                self.take_inventory(unit, 8, 1);
                self.farm.tiles[tile_index].fertilized_until_day =
                    tile.fertilized_until_day.max(day as i16 + 2);
            }
            53 => self.farm.tiles[tile_index] = Tile::default(),
            54 => self.farm.tiles[tile_index] = Tile::structure(TileKind::Coop),
            55 => self.farm.tiles[tile_index] = Tile::structure(TileKind::Pasture),
            56 => {
                self.take_inventory(unit, 0, 1);
                self.farm.tiles[tile_index].watered_or_fed = true;
            }
            57 => {
                self.farm.tiles[tile_index].fertilizer_available = false;
                self.add_inventory(unit, 8, 1);
            }
            58 => self.farm.tiles[tile_index].cared_today = true,
            _ => {}
        }
    }

    fn shed_total(&self) -> u16 {
        self.private.shed.iter().sum()
    }

    fn add_inventory(&mut self, unit: usize, item: usize, amount: u16) {
        if amount > 0 && self.private.inventories[unit][item] == 0 {
            let index = self.private.inventory_order[unit]
                .iter()
                .position(|&entry| entry == u8::MAX)
                .unwrap();
            self.private.inventory_order[unit][index] = item as u8;
        }
        self.private.inventories[unit][item] += amount;
    }

    fn take_inventory(&mut self, unit: usize, item: usize, amount: u16) {
        self.private.inventories[unit][item] -= amount;
        if self.private.inventories[unit][item] == 0 {
            remove_inventory_order(&mut self.private.inventory_order[unit], item);
        }
    }

    fn drop_inventory(&mut self, unit: usize) {
        for raw_item in self.private.inventory_order[unit] {
            if raw_item == u8::MAX {
                break;
            }
            let item = usize::from(raw_item);
            let room = self.config.shed_capacity.saturating_sub(self.shed_total());
            let amount = self.private.inventories[unit][item].min(room);
            self.private.shed[item] += amount;
            self.private.inventories[unit][item] = 0;
        }
        self.private.inventory_order[unit] = [u8::MAX; PRIVATE_ITEMS];
    }

    fn place_to_shed(&mut self, unit: usize, item: usize, requested: u16) {
        let available = self.private.inventories[unit][item];
        let room = self.config.shed_capacity.saturating_sub(self.shed_total());
        let quantity = requested.min(available).min(room);
        if quantity == 0 {
            return;
        }
        self.take_inventory(unit, item, quantity);
        self.private.shed[item] += quantity;
    }
}

#[derive(Clone)]
struct PolicyMarketLedger {
    money: i64,
    shed: [u16; PRIVATE_ITEMS],
    hires: u8,
    original_hires: u8,
    original_units: u8,
    extra_land: usize,
    inventory: [i32; PRODUCTS],
}

impl Game {
    pub fn new(seed: u64, config: GameConfig) -> Self {
        let spawn = default_spawn();
        let farms = std::array::from_fn(|_| Farm {
            money: config.starting_money,
            tiles: std::array::from_fn(|index| {
                let x = index % BOARD_SIZE;
                let y = index / BOARD_SIZE;
                if x < BOARD_SIZE / 2 && y < BOARD_SIZE / 2 {
                    Tile::default()
                } else {
                    Tile {
                        kind: TileKind::Locked,
                        max_lifespan_step: -1,
                        fertilized_until_day: -1,
                        ..Tile::default()
                    }
                }
            }),
            positions: std::array::from_fn(|_| spawn),
            units: 1,
            unlocked: 1,
            hires_today: 0,
        });
        let privates = std::array::from_fn(|_| PrivateState {
            shed: [0; PRIVATE_ITEMS],
            seeds: [0; CROPS],
            inventories: [[0; PRIVATE_ITEMS]; MAX_UNITS],
            inventory_order: [[u8::MAX; PRIVATE_ITEMS]; MAX_UNITS],
        });
        Self {
            seed,
            config,
            step: 0,
            done: false,
            farms,
            privates,
            market_inventory: [MARKET_I0; PRODUCTS],
            market_prices: std::array::from_fn(|item| market_price(item, MARKET_I0)),
            shops: [0; 8],
            shop_count: 0,
        }
    }

    pub fn step(&mut self, actions: &[CompactAction; PLAYERS]) -> StepResult {
        if self.done {
            return self.step_result();
        }
        let day = self.step / self.config.turns_per_day;
        for (player, action) in actions.iter().enumerate() {
            self.apply_unit_actions(player, action, day);
        }
        self.process_market(actions);
        self.town_consume(self.step);
        for player in 0..PLAYERS {
            self.decay_plants(player, self.step);
        }
        if (self.step + 1).is_multiple_of(self.config.turns_per_day) {
            self.end_of_day(day);
        }
        self.step += 1;
        if self.step >= self.config.episode_steps - 1 {
            self.done = true;
        }
        self.step_result()
    }

    fn step_result(&self) -> StepResult {
        StepResult {
            done: self.done,
            rewards: if self.done {
                [self.farms[0].money as f32, self.farms[1].money as f32]
            } else {
                [0.0, 0.0]
            },
            money: [self.farms[0].money as f32, self.farms[1].money as f32],
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_player(
        &self,
        player: usize,
        board: &mut [f32],
        globals: &mut [f32],
        critic: &mut [f32],
        units: &mut [f32],
        positions: &mut [i64],
        active: &mut [bool],
    ) {
        assert_eq!(board.len(), BOARD_CHANNELS * TILE_COUNT);
        assert_eq!(globals.len(), GLOBAL_FEATURES);
        assert_eq!(critic.len(), CRITIC_FEATURES);
        assert_eq!(units.len(), MAX_UNITS * UNIT_FEATURES);
        assert_eq!(positions.len(), MAX_UNITS * 2);
        assert_eq!(active.len(), MAX_UNITS);
        board.fill(0.0);
        globals.fill(0.0);
        critic.fill(0.0);
        units.fill(0.0);
        positions.fill(0);
        active.fill(false);

        let opponent = 1 - player;
        let day = self.step / self.config.turns_per_day;
        encode_farm(
            &self.farms[player],
            day,
            self.step,
            &mut board[..FARM_CHANNELS * TILE_COUNT],
        );
        encode_farm(
            &self.farms[opponent],
            day,
            self.step,
            &mut board[FARM_CHANNELS * TILE_COUNT..],
        );

        let hour = self.step % self.config.turns_per_day;
        let cycle =
            2.0 * std::f64::consts::PI * f64::from(hour) / f64::from(self.config.turns_per_day);
        let mut cursor = 0;
        let mut push = |value: f32| {
            globals[cursor] = value;
            cursor += 1;
        };
        push(f32::from(day) / 30.0);
        push(f32::from(hour) / f32::from(self.config.turns_per_day));
        push(f32::from(self.step) / 719.0);
        push(f32::from(719 - self.step) / 719.0);
        push(cycle.sin() as f32);
        push(cycle.cos() as f32);
        for index in [player, opponent] {
            let farm = &self.farms[index];
            push(money_feature(farm.money));
            push(farm.unlocked.count_ones() as f32 / 4.0);
            push(f32::from(farm.units - 1) / 15.0);
            push(f32::from(farm.hires_today) / 15.0);
        }
        let mut own_private = [0.0; 29];
        private_vector(&self.privates[player], &mut own_private);
        for value in own_private {
            push(value);
        }
        for inventory in self.market_inventory {
            push((inventory as f32 - 10_000.0) / 500.0);
        }
        for (item, price) in self.market_prices.iter().enumerate() {
            push(*price as f32 / (2.0 * MARKET_PARAMS[item].0 as f32));
        }
        for shop in 0..8 {
            let count = self.shops[..usize::from(self.shop_count)]
                .iter()
                .filter(|&&candidate| usize::from(candidate) == shop)
                .count();
            push(count as f32 / 8.0);
        }
        push(f32::from(self.shed_total(player)) / 100.0);
        let carried: u16 = self.privates[player]
            .inventories
            .iter()
            .flat_map(|inventory| inventory.iter())
            .copied()
            .sum();
        push(f32::from(carried) / 100.0);
        // Local self-play has the full nominal overage budget.
        push(1.0);
        debug_assert_eq!(cursor, GLOBAL_FEATURES);

        critic[..GLOBAL_FEATURES].copy_from_slice(globals);
        private_vector(
            &self.privates[opponent],
            (&mut critic[GLOBAL_FEATURES..]).try_into().unwrap(),
        );

        let farm = &self.farms[player];
        for unit in 0..usize::from(farm.units) {
            active[unit] = true;
            let position = farm.positions[unit];
            positions[unit * 2] = i64::from(position.0);
            positions[unit * 2 + 1] = i64::from(position.1);
            let row = &mut units[unit * UNIT_FEATURES..(unit + 1) * UNIT_FEATURES];
            row[0] = 1.0;
            row[1] = f32::from(unit == 0);
            row[2] = unit as f32 / 15.0;
            row[3] = f32::from(position.0) / 9.0;
            row[4] = f32::from(position.1) / 9.0;
            for item in 0..PRIVATE_ITEMS {
                row[5 + item] = f32::from(self.privates[player].inventories[unit][item]) / 32.0;
            }
        }
    }

    /// Encode one seat's structured token bundle, mirroring tokens.py exactly.
    ///
    /// Every continuous value is computed in f64 and truncated to f32 on
    /// store, reproducing the Python tokenizer's float64 -> float32 chain
    /// before the binding layer's final f16 staging cast. Categorical columns
    /// are small vocabulary indices staged as i8.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_player_structured(
        &self,
        player: usize,
        tile_categorical: &mut [i8],
        tile_continuous: &mut [f32],
        unit_categorical: &mut [i8],
        unit_continuous: &mut [f32],
        unit_active: &mut [bool],
        unit_tile_gather: &mut [i8],
        unit_tile_gather_valid: &mut [bool],
        products: &mut [f32],
        crops: &mut [f32],
        farms: &mut [f32],
        town: &mut [f32],
    ) {
        assert_eq!(tile_categorical.len(), TILE_TOKENS * TILE_CATEGORICAL);
        assert_eq!(tile_continuous.len(), TILE_TOKENS * TILE_CONTINUOUS);
        assert_eq!(unit_categorical.len(), MAX_UNITS * UNIT_CATEGORICAL);
        assert_eq!(unit_continuous.len(), MAX_UNITS * UNIT_CONTINUOUS);
        assert_eq!(unit_active.len(), MAX_UNITS);
        assert_eq!(unit_tile_gather.len(), MAX_UNITS * UNIT_GATHERS);
        assert_eq!(unit_tile_gather_valid.len(), MAX_UNITS * UNIT_GATHERS);
        assert_eq!(products.len(), PRODUCTS * PRODUCT_TOKEN_FIELDS);
        assert_eq!(crops.len(), CROPS * CROP_TOKEN_FIELDS);
        assert_eq!(farms.len(), PLAYERS * FARM_TOKEN_FIELDS);
        assert_eq!(town.len(), TOWN_TOKEN_FIELDS);
        tile_categorical.fill(0);
        tile_continuous.fill(0.0);
        unit_categorical.fill(0);
        unit_continuous.fill(0.0);
        unit_active.fill(false);
        unit_tile_gather.fill(0);
        unit_tile_gather_valid.fill(false);
        products.fill(0.0);
        crops.fill(0.0);
        farms.fill(0.0);
        town.fill(0.0);

        let opponent = 1 - player;
        let day = self.step / self.config.turns_per_day;
        encode_farm_structured(
            &self.farms[player],
            day,
            self.step,
            false,
            &mut tile_categorical[..TILE_COUNT * TILE_CATEGORICAL],
            &mut tile_continuous[..TILE_COUNT * TILE_CONTINUOUS],
        );
        encode_farm_structured(
            &self.farms[opponent],
            day,
            self.step,
            true,
            &mut tile_categorical[TILE_COUNT * TILE_CATEGORICAL..],
            &mut tile_continuous[TILE_COUNT * TILE_CONTINUOUS..],
        );

        let farm = &self.farms[player];
        let private = &self.privates[player];
        for unit in 0..usize::from(farm.units) {
            unit_active[unit] = true;
            let position = farm.positions[unit];
            let x = usize::from(position.0);
            let y = usize::from(position.1);
            let categorical =
                &mut unit_categorical[unit * UNIT_CATEGORICAL..(unit + 1) * UNIT_CATEGORICAL];
            categorical[0] = i8::from(unit != 0);
            categorical[1] = unit as i8;
            categorical[2] = y as i8;
            categorical[3] = x as i8;
            let continuous =
                &mut unit_continuous[unit * UNIT_CONTINUOUS..(unit + 1) * UNIT_CONTINUOUS];
            let mut total = 0.0f64;
            for (target, &count) in continuous[..PRIVATE_ITEMS]
                .iter_mut()
                .zip(&private.inventories[unit])
            {
                let held = f64::from(count);
                total += held;
                *target = (held / 32.0) as f32;
            }
            continuous[PRIVATE_ITEMS] = (total / 32.0) as f32;
            continuous[PRIVATE_ITEMS + 1] = f32::from(u8::from(is_shed_access(x, y)));
            const GATHER_DELTAS: [(i16, i16); UNIT_GATHERS] =
                [(0, 0), (0, -1), (0, 1), (1, 0), (-1, 0)];
            for (gather, (dx, dy)) in GATHER_DELTAS.iter().enumerate() {
                let nx = x as i16 + dx;
                let ny = y as i16 + dy;
                if (0..BOARD_SIZE as i16).contains(&nx) && (0..BOARD_SIZE as i16).contains(&ny) {
                    unit_tile_gather[unit * UNIT_GATHERS + gather] =
                        (ny * BOARD_SIZE as i16 + nx) as i8;
                    unit_tile_gather_valid[unit * UNIT_GATHERS + gather] = true;
                }
            }
        }

        let max_base_price = MARKET_PARAMS
            .iter()
            .map(|params| params.0)
            .fold(0.0, f64::max);
        for item in 0..PRODUCTS {
            let base = MARKET_PARAMS[item].0;
            let mut carried = 0.0f64;
            for inventory in &private.inventories[..usize::from(farm.units)] {
                carried += f64::from(inventory[item]);
            }
            let row = &mut products[item * PRODUCT_TOKEN_FIELDS..(item + 1) * PRODUCT_TOKEN_FIELDS];
            row[0] =
                ((f64::from(self.market_inventory[item]) - f64::from(MARKET_I0)) / 500.0) as f32;
            row[1] = (self.market_prices[item] as f64 / (2.0 * base)) as f32;
            row[2] = (base / max_base_price) as f32;
            row[3] = (f64::from(private.shed[item]) / f64::from(self.config.shed_capacity)) as f32;
            row[4] = (carried / f64::from(self.config.shed_capacity)) as f32;
        }
        let max_seed_cost = SEED_COST.iter().copied().max().unwrap() as f64;
        let max_yield_day = f64::from(*MAX_YIELD_DAY.iter().max().unwrap());
        let max_yield = f64::from(*CROP_MAX_HELD.iter().max().unwrap());
        for crop in 0..CROPS {
            let row = &mut crops[crop * CROP_TOKEN_FIELDS..(crop + 1) * CROP_TOKEN_FIELDS];
            row[0] = (SEED_COST[crop] as f64 / max_seed_cost) as f32;
            row[1] = (f64::from(private.seeds[crop]) / f64::from(self.config.shed_capacity)) as f32;
            row[2] = (f64::from(FIRST_YIELD[crop]) / max_yield_day) as f32;
            row[3] = (f64::from(MAX_YIELD_DAY[crop]) / max_yield_day) as f32;
            row[4] = (f64::from(CROP_MAX_HELD[crop]) / max_yield) as f32;
            row[5] = f32::from(u8::from(CROP_ONGOING[crop]));
        }
        for (slot, index) in [player, opponent].into_iter().enumerate() {
            let summary = &self.farms[index];
            let row = &mut farms[slot * FARM_TOKEN_FIELDS..(slot + 1) * FARM_TOKEN_FIELDS];
            row[0] = money_feature(summary.money);
            row[1] = (f64::from(summary.unlocked.count_ones()) / 4.0) as f32;
            row[2] = (f64::from(summary.units - 1) / f64::from(MAX_UNITS as u8 - 1)) as f32;
            row[3] = (f64::from(summary.hires_today) / f64::from(MAX_UNITS as u8 - 1)) as f32;
        }

        let hour = self.step % self.config.turns_per_day;
        let cycle =
            2.0 * std::f64::consts::PI * f64::from(hour) / f64::from(self.config.turns_per_day);
        let episode_days =
            f64::from(self.config.episode_steps) / f64::from(self.config.turns_per_day);
        let horizon = f64::from(self.config.episode_steps - 1);
        town[0] = (f64::from(day) / episode_days) as f32;
        town[1] = (f64::from(hour) / f64::from(self.config.turns_per_day)) as f32;
        town[2] = (f64::from(self.step) / horizon) as f32;
        town[3] = (f64::from(self.config.episode_steps - 1 - self.step) / horizon) as f32;
        town[4] = cycle.sin() as f32;
        town[5] = cycle.cos() as f32;
        for shop in 0..8 {
            let count = self.shops[..usize::from(self.shop_count)]
                .iter()
                .filter(|&&candidate| usize::from(candidate) == shop)
                .count();
            town[6 + shop] = (count as f64 / 8.0) as f32;
        }
    }

    /// Bank money plus the exact proceeds of liquidating every held product.
    ///
    /// Mirrors the engine's sell arithmetic unit by unit: each unit quotes at
    /// the current market inventory and a sale restocks the market only while
    /// the quote sits above the price floor.  Products in unit hands count at
    /// shed value, an optimistic bound: depositing needs shed room and one
    /// more turn.  Quotes and counts are integers, so the sum is exact in f64.
    pub fn liquidation_value(&self, player: usize) -> f64 {
        let mut held = [0_i64; PRODUCTS];
        for (count, &stored) in held.iter_mut().zip(&self.privates[player].shed) {
            *count = i64::from(stored);
        }
        let units = usize::from(self.farms[player].units);
        for inventory in &self.privates[player].inventories[..units] {
            for (count, &carried) in held.iter_mut().zip(&inventory[..PRODUCTS]) {
                *count += i64::from(carried);
            }
        }
        let mut value = self.farms[player].money as f64;
        for (item, &count) in held.iter().enumerate() {
            let mut market = self.market_inventory[item];
            for _ in 0..count {
                let price = market_price(item, market);
                value += price as f64;
                if price > PRICE_FLOOR {
                    market += 1;
                }
            }
        }
        value
    }

    /// Heuristic cost-basis credit for assets the market cannot buy back.
    ///
    /// Animals, seeds, planted crops, pending yields, and land have no exact
    /// cash value, so this credits fractions of engine cost (posted prices
    /// for pending yields) purely to smooth credit assignment across the
    /// invest-produce-sell loop: without it, self-play collapses into a
    /// never-spend tie equilibrium before harvests can pay back.  The
    /// terminal bank override keeps the objective exact regardless of these
    /// weights.
    pub fn illiquid_value(&self, player: usize) -> f64 {
        let mut value = 0.0;
        let units = usize::from(self.farms[player].units);
        for animal in 0..ANIMALS {
            let mut held = i64::from(self.privates[player].shed[PRODUCTS + animal]);
            for inventory in &self.privates[player].inventories[..units] {
                held += i64::from(inventory[PRODUCTS + animal]);
            }
            value += ILLIQUID_SHED_ANIMAL_CREDIT * held as f64 * ANIMAL_COST[animal] as f64;
        }
        #[allow(clippy::needless_range_loop)]
        for crop in 0..CROPS {
            value += ILLIQUID_SHED_SEED_CREDIT
                * f64::from(self.privates[player].seeds[crop])
                * SEED_COST[crop] as f64;
        }
        for tile in self.farms[player].tiles {
            if tile.has_animal {
                let animal = usize::from(tile.species);
                value += ILLIQUID_PLACED_ANIMAL_CREDIT * ANIMAL_COST[animal] as f64;
                value += ILLIQUID_PENDING_YIELD_CREDIT
                    * f64::from(tile.yield_units)
                    * self.market_prices[ANIMAL_PRODUCT[animal]] as f64;
            } else if tile.kind == TileKind::Plant {
                let crop = usize::from(tile.species);
                value += ILLIQUID_PLANTED_SEED_CREDIT * SEED_COST[crop] as f64;
                value += ILLIQUID_PENDING_YIELD_CREDIT
                    * f64::from(tile.yield_units)
                    * self.market_prices[crop] as f64;
            }
        }
        let extra = self.farms[player].unlocked.count_ones().saturating_sub(1) as usize;
        value += ILLIQUID_LAND_CREDIT * LAND_PRICES[..extra].iter().sum::<i64>() as f64;
        value
    }

    /// Bounded relative farm value from player zero's perspective.
    ///
    /// The dense shaping potential: an exact liquidation core (market product
    /// trades stay exactly potential-neutral and harvested-but-unsold output
    /// is credited at true sale proceeds) plus the heuristic cost-basis
    /// credit for illiquid assets.
    pub fn pair_potential(&self) -> f32 {
        relative_score(
            self.liquidation_value(0) + self.illiquid_value(0),
            self.liquidation_value(1) + self.illiquid_value(1),
        )
    }

    /// Bounded relative bank score, the quantity the engine actually scores.
    pub fn terminal_potential(&self) -> f32 {
        relative_score(self.farms[0].money as f64, self.farms[1].money as f64)
    }

    /// Post-step shaping potential: relative liquidation value mid-episode and
    /// relative bank at termination, so the telescoped shaped return equals
    /// the exact relative final bank score.
    pub fn post_step_potential(&self) -> f32 {
        if self.done {
            self.terminal_potential()
        } else {
            self.pair_potential()
        }
    }

    pub fn factor_masks(&self, player: usize, actions: &CompactAction) -> FactorMasks {
        let mut unit = [false; UNIT_MASK_VALUES];
        let mut market_kind = [false; MARKET_KIND_MASK_VALUES];
        let mut market_quantity = [false; MARKET_QUANTITY_MASK_VALUES];
        let mut unit_active = [false; MAX_UNITS];
        let mut market_active = [false; MAX_MARKET_ORDERS];
        let mut market_quantity_active = [false; MAX_MARKET_ORDERS];
        let day = self.step / self.config.turns_per_day;
        let mut unit_ledger = UnitLedger::from_game(self, player);
        let units = usize::from(self.farms[player].units);
        for unit_index in 0..MAX_UNITS {
            let row = &mut unit[unit_index * UNIT_ACTIONS..(unit_index + 1) * UNIT_ACTIONS];
            if unit_index >= units {
                row[0] = true;
                continue;
            }
            unit_active[unit_index] = true;
            for (action, valid) in row.iter_mut().enumerate() {
                *valid = unit_ledger.action_valid(unit_index, action as u8, day);
            }
            let selected = usize::from(actions.units[unit_index]);
            let applied = if selected < UNIT_ACTIONS && row[selected] {
                selected as u8
            } else {
                0
            };
            unit_ledger.apply_action(unit_index, applied, day);
        }

        let farm = &unit_ledger.farm;
        let mut ledger = PolicyMarketLedger {
            money: farm.money,
            shed: unit_ledger.private.shed,
            hires: farm.hires_today,
            original_hires: farm.hires_today,
            original_units: farm.units,
            extra_land: farm.unlocked.count_ones() as usize - 1,
            inventory: self.market_inventory,
        };
        let mut active = true;
        for slot in 0..MAX_MARKET_ORDERS {
            let kind_row = &mut market_kind[slot * MARKET_KINDS..(slot + 1) * MARKET_KINDS];
            let quantity_row =
                &mut market_quantity[slot * MARKET_QUANTITIES..(slot + 1) * MARKET_QUANTITIES];
            if !active {
                kind_row[0] = true;
                quantity_row[0] = true;
                continue;
            }
            market_active[slot] = true;
            fill_market_kind_mask(&unit_ledger.config, &ledger, kind_row);
            let selected = usize::from(actions.market_kinds[slot]);
            let kind = if selected < MARKET_KINDS && kind_row[selected] {
                selected as u8
            } else {
                0
            };
            if kind == 0 {
                quantity_row[0] = true;
                active = false;
                continue;
            }
            fill_market_quantity_mask(&unit_ledger.config, &ledger, kind, quantity_row);
            if kind >= 3 {
                market_quantity_active[slot] = true;
            }
            let selected_quantity = usize::from(actions.market_quantities[slot]);
            let quantity =
                if selected_quantity < MARKET_QUANTITIES && quantity_row[selected_quantity] {
                    selected_quantity as u16 + 1
                } else {
                    1
                };
            apply_policy_market_order(&unit_ledger.config, &mut ledger, kind, quantity);
        }
        FactorMasks {
            unit,
            market_kind,
            market_quantity,
            unit_active,
            market_active,
            market_quantity_active,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample_factors(
        &self,
        player: usize,
        unit_logits: &[f32],
        market_kind_logits: &[f32],
        market_quantity_context: &[f32],
        quantity_head: &QuantityHead<'_>,
        unit_draws: &[f32],
        market_kind_draws: &[f32],
        market_quantity_draws: &[f32],
        deterministic: bool,
        temperature: f32,
    ) -> SampledFactors {
        debug_assert_eq!(unit_logits.len(), MAX_UNITS * UNIT_ACTIONS);
        debug_assert_eq!(market_kind_logits.len(), MAX_MARKET_ORDERS * MARKET_KINDS);
        debug_assert_eq!(
            market_quantity_context.len(),
            MAX_MARKET_ORDERS * quantity_head.rank
        );
        debug_assert_eq!(
            quantity_head.kind_gate.len(),
            MARKET_KINDS * quantity_head.rank
        );
        debug_assert_eq!(
            quantity_head.values.len(),
            MARKET_QUANTITIES * quantity_head.rank
        );
        debug_assert_eq!(quantity_head.bias.len(), MARKET_KINDS * MARKET_QUANTITIES);
        let mut action = CompactAction::default();
        let mut unit_masks = [false; UNIT_MASK_VALUES];
        let mut market_kind_masks = [false; MARKET_KIND_MASK_VALUES];
        let mut market_quantity_masks = [false; MARKET_QUANTITY_MASK_VALUES];
        let mut unit_active = [false; MAX_UNITS];
        let mut market_active = [false; MAX_MARKET_ORDERS];
        let mut market_quantity_active = [false; MAX_MARKET_ORDERS];
        let mut unit_logprobs = [0.0; MAX_UNITS];
        let mut market_kind_logprobs = [0.0; MAX_MARKET_ORDERS];
        let mut market_quantity_logprobs = [0.0; MAX_MARKET_ORDERS];
        let mut unit_entropies = [0.0; MAX_UNITS];
        let mut market_kind_entropies = [0.0; MAX_MARKET_ORDERS];
        let mut market_quantity_entropies = [0.0; MAX_MARKET_ORDERS];
        let mut entropy_sum = 0.0f32;
        let mut component_count = 0usize;
        let day = self.step / self.config.turns_per_day;
        let mut unit_ledger = UnitLedger::from_game(self, player);
        let active_units = usize::from(self.farms[player].units);
        for unit in 0..MAX_UNITS {
            let mask = &mut unit_masks[unit * UNIT_ACTIONS..(unit + 1) * UNIT_ACTIONS];
            if unit >= active_units {
                mask[0] = true;
            } else {
                unit_active[unit] = true;
                for (candidate, valid) in mask.iter_mut().enumerate() {
                    *valid = unit_ledger.action_valid(unit, candidate as u8, day);
                }
            }
            let (selected, logprob, entropy) = sample_categorical(
                &unit_logits[unit * UNIT_ACTIONS..(unit + 1) * UNIT_ACTIONS],
                mask,
                deterministic,
                temperature,
                unit_draws[unit],
            );
            action.units[unit] = selected as u8;
            unit_logprobs[unit] = logprob;
            unit_entropies[unit] = entropy;
            if unit_active[unit] {
                entropy_sum += entropy;
                component_count += 1;
                unit_ledger.apply_action(unit, selected as u8, day);
            }
        }

        let farm = &unit_ledger.farm;
        let mut ledger = PolicyMarketLedger {
            money: farm.money,
            shed: unit_ledger.private.shed,
            hires: farm.hires_today,
            original_hires: farm.hires_today,
            original_units: farm.units,
            extra_land: farm.unlocked.count_ones() as usize - 1,
            inventory: self.market_inventory,
        };
        let mut still_active = true;
        let mut quantity_logits = [0.0f32; MARKET_QUANTITIES];
        for slot in 0..MAX_MARKET_ORDERS {
            let kind_mask = &mut market_kind_masks[slot * MARKET_KINDS..(slot + 1) * MARKET_KINDS];
            let quantity_mask = &mut market_quantity_masks
                [slot * MARKET_QUANTITIES..(slot + 1) * MARKET_QUANTITIES];
            if still_active {
                market_active[slot] = true;
                fill_market_kind_mask(&unit_ledger.config, &ledger, kind_mask);
            } else {
                kind_mask[0] = true;
            }
            let (kind, kind_logprob, kind_entropy) = sample_categorical(
                &market_kind_logits[slot * MARKET_KINDS..(slot + 1) * MARKET_KINDS],
                kind_mask,
                deterministic,
                temperature,
                market_kind_draws[slot],
            );
            action.market_kinds[slot] = kind as u8;
            market_kind_logprobs[slot] = kind_logprob;
            market_kind_entropies[slot] = kind_entropy;
            if market_active[slot] {
                entropy_sum += kind_entropy;
                component_count += 1;
            }
            if !still_active || kind == 0 {
                quantity_mask[0] = true;
                still_active = false;
                continue;
            }
            fill_market_quantity_mask(&unit_ledger.config, &ledger, kind as u8, quantity_mask);
            if kind < 3 {
                // HIRE and BUY_LAND have no quantity factor. Preserve the
                // conventional zero category/log-probability outputs without
                // touching the low-rank quantity head or consuming its draw.
                action.market_quantities[slot] = 0;
                apply_policy_market_order(&unit_ledger.config, &mut ledger, kind as u8, 1);
                continue;
            }
            market_quantity_active[slot] = true;
            score_quantities(
                &market_quantity_context
                    [slot * quantity_head.rank..(slot + 1) * quantity_head.rank],
                kind,
                quantity_head,
                &mut quantity_logits,
            );
            let (quantity, quantity_logprob, quantity_entropy) = sample_categorical(
                &quantity_logits,
                quantity_mask,
                deterministic,
                temperature,
                market_quantity_draws[slot],
            );
            action.market_quantities[slot] = quantity as u8;
            market_quantity_logprobs[slot] = quantity_logprob;
            market_quantity_entropies[slot] = quantity_entropy;
            if market_quantity_active[slot] {
                entropy_sum += quantity_entropy;
                component_count += 1;
            }
            apply_policy_market_order(
                &unit_ledger.config,
                &mut ledger,
                kind as u8,
                quantity as u16 + 1,
            );
        }

        SampledFactors {
            action,
            masks: FactorMasks {
                unit: unit_masks,
                market_kind: market_kind_masks,
                market_quantity: market_quantity_masks,
                unit_active,
                market_active,
                market_quantity_active,
            },
            unit_logprobs,
            market_kind_logprobs,
            market_quantity_logprobs,
            unit_entropies,
            market_kind_entropies,
            market_quantity_entropies,
            mean_entropy: entropy_sum / component_count.max(1) as f32,
        }
    }

    fn apply_unit_actions(&mut self, player: usize, actions: &CompactAction, day: u16) {
        let units = usize::from(self.farms[player].units);
        for unit in 0..units {
            // `step_factors` receives the policy's pre-compile factors. The Python
            // compiler reserves seeds sequentially, so later excess plants become
            // PASS rather than triggering the raw interpreter's all-or-none rule.
            let selected = actions.units[unit];
            let action = if self.unit_action_valid(player, unit, selected, day) {
                selected
            } else {
                0
            };
            self.apply_unit_action(player, unit, action, day);
        }
    }

    pub fn unit_action_valid(&self, player: usize, unit: usize, action: u8, day: u16) -> bool {
        if action >= UNIT_ACTIONS as u8 || unit >= usize::from(self.farms[player].units) {
            return false;
        }
        if action == 0 {
            return true;
        }
        let position = self.farms[player].positions[unit];
        let x = usize::from(position.0);
        let y = usize::from(position.1);
        if let Some((dx, dy)) = move_delta(action) {
            let nx = x as i16 + dx;
            let ny = y as i16 + dy;
            return (0..BOARD_SIZE as i16).contains(&nx) && (0..BOARD_SIZE as i16).contains(&ny);
        }
        let at_shed = is_shed_access(x, y);
        if action == 5 {
            return at_shed
                && self.shed_total(player) < self.config.shed_capacity
                && self.privates[player].inventories[unit]
                    .iter()
                    .any(|&quantity| quantity > 0);
        }
        if let Some((item, quantity)) = pickup_spec(action) {
            return at_shed && self.privates[player].shed[item] >= quantity;
        }
        let tile = self.farms[player].tiles[y * BOARD_SIZE + x];
        if let Some(animal) = place_animal(action) {
            let has_animal = self.privates[player].inventories[unit][9 + animal] > 0;
            let installs_animal = tile.kind == animal_structure(animal) && !tile.has_animal;
            let deposits_animal = at_shed && self.shed_total(player) < self.config.shed_capacity;
            return has_animal && (installs_animal || deposits_animal);
        }
        if tile.kind == TileKind::Locked {
            return false;
        }
        if let Some(crop) = unit_plant_crop(action) {
            return tile.kind == TileKind::Empty && self.privates[player].seeds[crop] > 0;
        }
        match action {
            50 => tile.kind == TileKind::Plant && !tile.watered_or_fed,
            51 if tile.kind == TileKind::Plant => {
                tile.yield_units > 0
                    && day.saturating_sub(tile.origin_day) >= FIRST_YIELD[usize::from(tile.species)]
            }
            51 => tile.has_animal && tile.yield_units > 0,
            52 => {
                tile.kind == TileKind::Plant
                    && self.privates[player].inventories[unit][8] > 0
                    && tile.fertilized_until_day < day as i16 + 2
            }
            53 => tile.kind != TileKind::Empty && !tile.has_animal,
            54 | 55 => tile.kind == TileKind::Empty,
            56 => {
                tile.has_animal
                    && !tile.watered_or_fed
                    && self.privates[player].inventories[unit][0] > 0
            }
            57 => tile.has_animal && tile.fertilizer_available,
            58 => tile.has_animal && !tile.cared_today,
            _ => false,
        }
    }

    fn apply_unit_action(&mut self, player: usize, unit: usize, action: u8, day: u16) {
        if action >= UNIT_ACTIONS as u8 || unit >= usize::from(self.farms[player].units) {
            return;
        }
        let position = self.farms[player].positions[unit];
        let x = usize::from(position.0);
        let y = usize::from(position.1);
        if let Some((dx, dy)) = move_delta(action) {
            let nx = x as i16 + dx;
            let ny = y as i16 + dy;
            if (0..BOARD_SIZE as i16).contains(&nx) && (0..BOARD_SIZE as i16).contains(&ny) {
                self.farms[player].positions[unit] = Position(nx as u8, ny as u8);
            }
            return;
        }
        if action == 0 {
            return;
        }
        let tile_index = y * BOARD_SIZE + x;

        // DROP and PICKUP precede the locked-tile guard in the official engine.
        if action == 5 {
            if is_shed_access(x, y) {
                self.drop_inventory(player, unit);
            }
            return;
        }
        if let Some((item, requested)) = pickup_spec(action) {
            if is_shed_access(x, y) {
                let quantity = self.privates[player].shed[item].min(requested);
                self.privates[player].shed[item] -= quantity;
                self.add_inventory(player, unit, item, quantity);
            }
            return;
        }
        if let Some(animal) = place_animal(action) {
            let tile = self.farms[player].tiles[tile_index];
            let private_item = 9 + animal;
            if tile.kind == animal_structure(animal)
                && !tile.has_animal
                && self.privates[player].inventories[unit][private_item] > 0
            {
                self.take_inventory(player, unit, private_item, 1);
                self.farms[player].tiles[tile_index] = Tile::animal(animal, day);
            } else if is_shed_access(x, y) {
                self.place_to_shed(player, unit, private_item, 1);
            }
            return;
        }
        let tile = self.farms[player].tiles[tile_index];
        if tile.kind == TileKind::Locked {
            return;
        }
        if let Some(crop) = unit_plant_crop(action) {
            if tile.kind == TileKind::Empty && self.privates[player].seeds[crop] > 0 {
                self.privates[player].seeds[crop] -= 1;
                self.farms[player].tiles[tile_index] =
                    Tile::plant(crop, day, self.config.turns_per_day);
            }
            return;
        }
        match action {
            50 => {
                if tile.kind != TileKind::Plant || tile.watered_or_fed {
                    return;
                }
                let crop = usize::from(tile.species);
                let mutable = &mut self.farms[player].tiles[tile_index];
                mutable.watered_or_fed = true;
                if !CROP_ONGOING[crop] {
                    let age = day - tile.origin_day;
                    let start = MAX_YIELD_DAY[crop].div_ceil(2);
                    if (start..=MAX_YIELD_DAY[crop]).contains(&age) {
                        let bonus = if tile.fertilized_until_day >= day as i16 {
                            2
                        } else {
                            1
                        };
                        mutable.yield_units =
                            CROP_MAX_HELD[crop].min(mutable.yield_units.saturating_add(bonus));
                    }
                }
            }
            51 => self.harvest(player, unit, tile_index, day),
            52 => {
                if tile.kind == TileKind::Plant && self.take_inventory(player, unit, 8, 1) {
                    self.farms[player].tiles[tile_index].fertilized_until_day =
                        tile.fertilized_until_day.max(day as i16 + 2);
                }
            }
            53 => {
                if tile.kind != TileKind::Empty && !tile.has_animal {
                    self.farms[player].tiles[tile_index] = Tile::default();
                }
            }
            54 if tile.kind == TileKind::Empty => {
                self.farms[player].tiles[tile_index] = Tile::structure(TileKind::Coop);
            }
            55 if tile.kind == TileKind::Empty => {
                self.farms[player].tiles[tile_index] = Tile::structure(TileKind::Pasture);
            }
            56 => {
                if tile.has_animal
                    && !tile.watered_or_fed
                    && self.take_inventory(player, unit, 0, 1)
                {
                    self.farms[player].tiles[tile_index].watered_or_fed = true;
                }
            }
            57 if tile.has_animal && tile.fertilizer_available => {
                self.farms[player].tiles[tile_index].fertilizer_available = false;
                self.add_inventory(player, unit, 8, 1);
            }
            58 if tile.has_animal && !tile.cared_today => {
                self.farms[player].tiles[tile_index].cared_today = true;
            }
            _ => {}
        }
    }

    fn harvest(&mut self, player: usize, unit: usize, tile_index: usize, day: u16) {
        let tile = self.farms[player].tiles[tile_index];
        if tile.yield_units == 0 {
            return;
        }
        if tile.kind == TileKind::Plant {
            let crop = usize::from(tile.species);
            if day - tile.origin_day < FIRST_YIELD[crop] {
                return;
            }
            self.add_inventory(player, unit, crop, u16::from(tile.yield_units));
            if CROP_ONGOING[crop] {
                self.farms[player].tiles[tile_index].yield_units = 0;
            } else {
                self.farms[player].tiles[tile_index] = Tile::default();
            }
        } else if tile.has_animal {
            let product = ANIMAL_PRODUCT[usize::from(tile.species)];
            self.add_inventory(player, unit, product, u16::from(tile.yield_units));
            self.farms[player].tiles[tile_index].yield_units = 0;
        }
    }

    #[inline]
    fn add_inventory(&mut self, player: usize, unit: usize, item: usize, n: u16) {
        if n == 0 {
            return;
        }
        if self.privates[player].inventories[unit][item] == 0 {
            let order = &mut self.privates[player].inventory_order[unit];
            let insertion = order.iter().position(|&entry| entry == u8::MAX).unwrap();
            order[insertion] = item as u8;
        }
        self.privates[player].inventories[unit][item] += n;
    }

    #[inline]
    fn take_inventory(&mut self, player: usize, unit: usize, item: usize, n: u16) -> bool {
        let quantity = &mut self.privates[player].inventories[unit][item];
        if *quantity < n {
            return false;
        }
        *quantity -= n;
        if *quantity == 0 {
            remove_inventory_order(&mut self.privates[player].inventory_order[unit], item);
        }
        true
    }

    fn shed_total(&self, player: usize) -> u16 {
        self.privates[player].shed.iter().sum()
    }

    fn drop_inventory(&mut self, player: usize, unit: usize) {
        let order = self.privates[player].inventory_order[unit];
        for raw_item in order {
            if raw_item == u8::MAX {
                break;
            }
            let item = usize::from(raw_item);
            let quantity = self.privates[player].inventories[unit][item];
            let room = self
                .config
                .shed_capacity
                .saturating_sub(self.shed_total(player));
            let take = quantity.min(room);
            self.privates[player].shed[item] += take;
            // Official DROP discards overflow too.
            self.privates[player].inventories[unit][item] = 0;
        }
        self.privates[player].inventory_order[unit] = [u8::MAX; PRIVATE_ITEMS];
    }

    fn place_to_shed(&mut self, player: usize, unit: usize, item: usize, requested: u16) {
        let available = self.privates[player].inventories[unit][item];
        let room = self
            .config
            .shed_capacity
            .saturating_sub(self.shed_total(player));
        let take = requested.min(available).min(room);
        self.privates[player].inventories[unit][item] -= take;
        if self.privates[player].inventories[unit][item] == 0 {
            remove_inventory_order(&mut self.privates[player].inventory_order[unit], item);
        }
        self.privates[player].shed[item] += take;
    }

    fn process_market(&mut self, actions: &[CompactAction; PLAYERS]) {
        let mut queue_active = [true; PLAYERS];
        for slot in 0..MAX_MARKET_ORDERS {
            let mut orders: [Option<MarketOrder>; PLAYERS] = std::array::from_fn(|player| {
                if !queue_active[player] || actions[player].market_kinds[slot] == 0 {
                    queue_active[player] = false;
                    None
                } else {
                    parse_order(
                        actions[player].market_kinds[slot],
                        actions[player].market_quantities[slot],
                    )
                }
            });
            #[allow(clippy::needless_range_loop)]
            for player in 0..PLAYERS {
                let Some(order) = orders[player] else {
                    continue;
                };
                match order.kind {
                    1 => {
                        self.hire(player);
                        orders[player] = None;
                    }
                    2 => {
                        self.buy_land(player);
                        orders[player] = None;
                    }
                    _ => {}
                }
            }
            loop {
                let mut quoted = [None; PLAYERS];
                for player in 0..PLAYERS {
                    let Some(order) = orders[player] else {
                        continue;
                    };
                    if order.remaining == 0 {
                        continue;
                    }
                    let price = match order.kind {
                        3..=7 => SEED_COST[order.item],
                        8..=9 => market_price(order.item, self.market_inventory[order.item] - 1),
                        10..=12 => ANIMAL_COST[order.item],
                        13..=21 => market_price(order.item, self.market_inventory[order.item]),
                        _ => {
                            orders[player] = None;
                            continue;
                        }
                    };
                    quoted[player] = Some((order, price));
                }
                if quoted.iter().all(Option::is_none) {
                    break;
                }
                let mut committed = false;
                for player in 0..PLAYERS {
                    let Some((order, price)) = quoted[player] else {
                        continue;
                    };
                    if self.commit_market_unit(player, order, price) {
                        if let Some(state) = &mut orders[player] {
                            state.remaining -= 1;
                        }
                        committed = true;
                    } else {
                        orders[player] = None;
                    }
                }
                if !committed {
                    break;
                }
            }
            self.refresh_prices();
        }
    }

    fn commit_market_unit(&mut self, player: usize, order: MarketOrder, price: i64) -> bool {
        match order.kind {
            3..=7 => {
                if self.farms[player].money < price {
                    return false;
                }
                self.farms[player].money -= price;
                self.privates[player].seeds[order.item] += 1;
            }
            8..=9 => {
                if self.farms[player].money < price
                    || self.shed_total(player) >= self.config.shed_capacity
                {
                    return false;
                }
                self.farms[player].money -= price;
                self.privates[player].shed[order.item] += 1;
                self.market_inventory[order.item] -= 1;
            }
            10..=12 => {
                if self.farms[player].money < price
                    || self.shed_total(player) >= self.config.shed_capacity
                {
                    return false;
                }
                self.farms[player].money -= price;
                self.privates[player].shed[9 + order.item] += 1;
            }
            13..=21 => {
                if self.privates[player].shed[order.item] == 0 {
                    return false;
                }
                self.privates[player].shed[order.item] -= 1;
                self.farms[player].money += price;
                if price > PRICE_FLOOR {
                    self.market_inventory[order.item] += 1;
                }
            }
            _ => return false,
        }
        true
    }

    fn hire(&mut self, player: usize) {
        let hires = self.farms[player].hires_today;
        if usize::from(self.farms[player].units) >= MAX_UNITS {
            return;
        }
        let cost = self.config.farm_hand_cost_mult * fib(hires);
        if self.farms[player].money < cost {
            return;
        }
        self.farms[player].money -= cost;
        self.farms[player].hires_today += 1;
        let position = spawn_hand(&self.farms[player]);
        let index = usize::from(self.farms[player].units);
        self.farms[player].positions[index] = position;
        self.farms[player].units += 1;
        self.privates[player].inventories[index] = [0; PRIVATE_ITEMS];
        self.privates[player].inventory_order[index] = [u8::MAX; PRIVATE_ITEMS];
    }

    fn buy_land(&mut self, player: usize) {
        let extra = self.farms[player].unlocked.count_ones() as usize - 1;
        if extra >= 3 || self.farms[player].money < LAND_PRICES[extra] {
            return;
        }
        self.farms[player].money -= LAND_PRICES[extra];
        let quadrant = extra + 1;
        self.farms[player].unlocked |= 1 << quadrant;
        for y in 0..BOARD_SIZE {
            for x in 0..BOARD_SIZE {
                if quadrant_of(x, y) == quadrant {
                    let tile = &mut self.farms[player].tiles[y * BOARD_SIZE + x];
                    if tile.kind == TileKind::Locked {
                        *tile = Tile::default();
                    }
                }
            }
        }
    }

    fn town_consume(&mut self, step: u16) {
        if step.is_multiple_of(self.config.shop_sell_interval) {
            for &shop in &self.shops[..usize::from(self.shop_count)] {
                let products = SHOP_PRODUCTS[usize::from(shop)];
                let multiplier = if products.len() == 1 { 2 } else { 1 };
                for &item in products {
                    self.market_inventory[item] -= multiplier;
                }
            }
        }
        if step.is_multiple_of(self.config.town_center_sell_interval) {
            for item in 0..PRODUCTS - 1 {
                self.market_inventory[item] -= 1;
            }
        }
        self.refresh_prices();
    }

    fn refresh_prices(&mut self) {
        for item in 0..PRODUCTS {
            self.market_prices[item] = market_price(item, self.market_inventory[item]);
        }
    }

    fn decay_plants(&mut self, player: usize, step: u16) {
        for tile in &mut self.farms[player].tiles {
            if tile.kind != TileKind::Plant
                || tile.max_lifespan_step < 0
                || (step as i16) < tile.max_lifespan_step
                || (step as i16 - tile.max_lifespan_step) % 2 != 0
            {
                continue;
            }
            tile.yield_units = tile.yield_units.saturating_sub(1);
            if tile.yield_units == 0 {
                *tile = Tile::structure(TileKind::Weed);
            }
        }
    }

    fn end_of_day(&mut self, day: u16) {
        let mixed_seed = (self.seed * 1_000_003) ^ u64::from(day);
        let mut rng = PyRandom::seed_u64(mixed_seed);
        for player in 0..PLAYERS {
            self.daily_refresh_plants(player, day);
            self.daily_refresh_animals(player, day);
            for tile in &mut self.farms[player].tiles {
                if tile.kind == TileKind::Empty && rng.random() < self.config.weed_spawn_chance {
                    *tile = Tile::structure(TileKind::Weed);
                }
            }
            for unit in 0..usize::from(self.farms[player].units) {
                self.drop_inventory(player, unit);
            }
            self.farms[player].positions.fill(default_spawn());
            self.farms[player].units = 1;
            self.farms[player].hires_today = 0;
            self.privates[player].inventories = [[0; PRIVATE_ITEMS]; MAX_UNITS];
            self.privates[player].inventory_order = [[u8::MAX; PRIVATE_ITEMS]; MAX_UNITS];
        }
        let next_day = day + 1;
        if next_day > 0
            && next_day.is_multiple_of(self.config.shop_unlock_interval)
            && usize::from(self.shop_count) < self.shops.len()
        {
            self.shops[usize::from(self.shop_count)] = rng.randbelow(8) as u8;
            self.shop_count += 1;
        }
    }

    fn daily_refresh_plants(&mut self, player: usize, day: u16) {
        let next_day = day + 1;
        for tile in &mut self.farms[player].tiles {
            if tile.kind != TileKind::Plant {
                continue;
            }
            let watered = tile.watered_or_fed;
            tile.consecutive_unmet = if watered {
                0
            } else {
                tile.consecutive_unmet + 1
            };
            tile.watered_or_fed = false;
            if tile.consecutive_unmet >= 2 {
                *tile = Tile::structure(TileKind::Weed);
                continue;
            }
            let crop = usize::from(tile.species);
            if !CROP_ONGOING[crop] {
                continue;
            }
            let days_since_first =
                next_day as i32 - tile.origin_day as i32 - FIRST_YIELD[crop] as i32;
            if days_since_first < 0 || days_since_first % i32::from(CROP_INTERVAL[crop]) != 0 {
                continue;
            }
            let production_count = days_since_first / i32::from(CROP_INTERVAL[crop]) + 1;
            if production_count > i32::from(CROP_MAX_HELD[crop]) {
                continue;
            }
            let fertilized = watered && tile.fertilized_until_day >= day as i16;
            tile.yield_units =
                CROP_MAX_HELD[crop].min(tile.yield_units.saturating_add(if fertilized {
                    2
                } else {
                    1
                }));
            if production_count == i32::from(CROP_MAX_HELD[crop]) {
                tile.max_lifespan_step = ((next_day + 1) * self.config.turns_per_day) as i16;
            }
        }
    }

    fn daily_refresh_animals(&mut self, player: usize, day: u16) {
        let next_day = day + 1;
        for tile in &mut self.farms[player].tiles {
            if !tile.has_animal {
                continue;
            }
            let fed = tile.watered_or_fed;
            tile.consecutive_unmet = if fed { 0 } else { tile.consecutive_unmet + 1 };
            if tile.consecutive_unmet >= 2 {
                *tile = Tile::structure(animal_structure(usize::from(tile.species)));
                continue;
            }
            let animal = usize::from(tile.species);
            let days_since_first =
                next_day as i32 - tile.origin_day as i32 - ANIMAL_FIRST_YIELD[animal] as i32;
            if days_since_first >= 0 && days_since_first % i32::from(ANIMAL_INTERVAL[animal]) == 0 {
                let bonus = if fed { tile.pending_care_bonus } else { 0 };
                tile.yield_units =
                    ANIMAL_MAX_HELD[animal].min(tile.yield_units.saturating_add(1 + bonus));
                tile.pending_care_bonus = 0;
            }
            if tile.cared_today && fed {
                tile.pending_care_bonus += 1;
            }
            tile.fertilizer_available = true;
            tile.watered_or_fed = false;
            tile.cared_today = false;
        }
    }
}

fn parse_order(kind: u8, quantity_index: u8) -> Option<MarketOrder> {
    if kind == 0 || kind >= MARKET_KINDS as u8 {
        return None;
    }
    let quantity = u16::from(quantity_index.min(99)) + 1;
    let item = match kind {
        3..=7 => usize::from(kind - 3),
        8 => 0,
        9 => 8,
        10..=12 => usize::from(kind - 10),
        13..=21 => usize::from(kind - 13),
        _ => 0,
    };
    Some(MarketOrder {
        kind,
        item,
        remaining: if matches!(kind, 1 | 2) { 0 } else { quantity },
    })
}

fn fill_market_kind_mask(config: &GameConfig, ledger: &PolicyMarketLedger, mask: &mut [bool]) {
    debug_assert_eq!(mask.len(), MARKET_KINDS);
    mask.fill(false);
    mask[0] = true;
    let added_hires = ledger.hires.saturating_sub(ledger.original_hires);
    mask[1] = usize::from(ledger.original_units) + usize::from(added_hires) < MAX_UNITS
        && ledger.money >= config.farm_hand_cost_mult * fib(ledger.hires);
    mask[2] =
        ledger.extra_land < LAND_PRICES.len() && ledger.money >= LAND_PRICES[ledger.extra_land];
    for crop in 0..CROPS {
        mask[3 + crop] = ledger.money >= SEED_COST[crop];
    }
    let room = config
        .shed_capacity
        .saturating_sub(ledger.shed.iter().sum());
    for (kind, item) in [(8, 0), (9, 8)] {
        let quote = market_price(item, ledger.inventory[item] - 1);
        mask[kind] = room > 0 && ledger.money >= quote;
    }
    for animal in 0..ANIMALS {
        mask[10 + animal] = room > 0 && ledger.money >= ANIMAL_COST[animal];
    }
    for product in 0..PRODUCTS {
        mask[13 + product] = ledger.shed[product] > 0;
    }
}

fn fill_market_quantity_mask(
    config: &GameConfig,
    ledger: &PolicyMarketLedger,
    kind: u8,
    mask: &mut [bool],
) {
    debug_assert_eq!(mask.len(), MARKET_QUANTITIES);
    mask.fill(false);
    if kind < 3 {
        mask[0] = true;
        return;
    }
    let room = config
        .shed_capacity
        .saturating_sub(ledger.shed.iter().sum());
    let maximum = match kind {
        3..=7 => (ledger.money / SEED_COST[usize::from(kind - 3)]).max(0) as usize,
        8 | 9 => {
            let item = if kind == 8 { 0 } else { 8 };
            let mut inventory = ledger.inventory[item];
            let mut money = ledger.money;
            let mut count = 0usize;
            while count < usize::from(room) {
                let quote = market_price(item, inventory - 1);
                if money < quote {
                    break;
                }
                money -= quote;
                inventory -= 1;
                count += 1;
            }
            count
        }
        10..=12 => {
            let animal = usize::from(kind - 10);
            usize::from(room).min((ledger.money / ANIMAL_COST[animal]).max(0) as usize)
        }
        13..=21 => usize::from(ledger.shed[usize::from(kind - 13)]),
        _ => 0,
    };
    mask.iter_mut()
        .take(maximum.min(MARKET_QUANTITIES))
        .for_each(|valid| *valid = true);
}

fn apply_policy_market_order(
    config: &GameConfig,
    ledger: &mut PolicyMarketLedger,
    kind: u8,
    quantity: u16,
) {
    match kind {
        1 => {
            ledger.money -= config.farm_hand_cost_mult * fib(ledger.hires);
            ledger.hires += 1;
        }
        2 => {
            ledger.money -= LAND_PRICES[ledger.extra_land];
            ledger.extra_land += 1;
        }
        3..=7 => {
            ledger.money -= SEED_COST[usize::from(kind - 3)] * i64::from(quantity);
        }
        8 | 9 => {
            let item = if kind == 8 { 0 } else { 8 };
            for _ in 0..quantity {
                let quote = market_price(item, ledger.inventory[item] - 1);
                if ledger.money < quote || ledger.shed.iter().sum::<u16>() >= config.shed_capacity {
                    break;
                }
                ledger.money -= quote;
                ledger.shed[item] += 1;
                ledger.inventory[item] -= 1;
            }
        }
        10..=12 => {
            let animal = usize::from(kind - 10);
            let amount = quantity.min(
                config
                    .shed_capacity
                    .saturating_sub(ledger.shed.iter().sum()),
            );
            ledger.money -= ANIMAL_COST[animal] * i64::from(amount);
            ledger.shed[PRODUCTS + animal] += amount;
        }
        13..=21 => {
            let item = usize::from(kind - 13);
            for _ in 0..quantity {
                if ledger.shed[item] == 0 {
                    break;
                }
                let quote = market_price(item, ledger.inventory[item]);
                ledger.shed[item] -= 1;
                ledger.money += quote;
                if quote > PRICE_FLOOR {
                    ledger.inventory[item] += 1;
                }
            }
        }
        _ => {}
    }
}

fn score_quantities(
    context: &[f32],
    kind: usize,
    head: &QuantityHead<'_>,
    output: &mut [f32; MARKET_QUANTITIES],
) {
    #[allow(clippy::needless_range_loop)]
    for quantity in 0..MARKET_QUANTITIES {
        let mut score = head.bias[kind * MARKET_QUANTITIES + quantity];
        #[allow(clippy::needless_range_loop)]
        for rank in 0..head.rank {
            let feature = context[rank] * (1.0 + head.kind_gate[kind * head.rank + rank]);
            score += feature * head.values[quantity * head.rank + rank];
        }
        output[quantity] = score;
    }
}

fn sample_categorical(
    logits: &[f32],
    mask: &[bool],
    deterministic: bool,
    temperature: f32,
    draw: f32,
) -> (usize, f32, f32) {
    debug_assert_eq!(logits.len(), mask.len());
    debug_assert!(logits.len() <= MARKET_QUANTITIES);
    debug_assert!(mask.iter().any(|&valid| valid));
    let temperature = temperature.max(1e-4);
    let mut shifted = [0.0f32; MARKET_QUANTITIES];
    let mut maximum = f32::NEG_INFINITY;
    let mut argmax = 0usize;
    for index in 0..logits.len() {
        if mask[index] {
            let value = logits[index] / temperature;
            if value > maximum {
                maximum = value;
                argmax = index;
            }
        }
    }
    let mut total = 0.0f32;
    for index in 0..logits.len() {
        if mask[index] {
            shifted[index] = (logits[index] / temperature - maximum).exp();
            total += shifted[index];
        }
    }
    let mut selected = argmax;
    if !deterministic {
        let mut cumulative = 0.0f32;
        selected = mask.iter().rposition(|&valid| valid).unwrap();
        for (index, probability) in shifted[..logits.len()].iter().enumerate() {
            if !mask[index] {
                continue;
            }
            cumulative += *probability / total;
            if draw <= cumulative {
                selected = index;
                break;
            }
        }
    }
    let selected_probability = (shifted[selected] / total).max(f32::MIN_POSITIVE);
    let mut entropy = 0.0f32;
    for raw_probability in &shifted[..logits.len()] {
        let probability = *raw_probability / total;
        if probability > 0.0 {
            entropy -= probability * probability.max(f32::MIN_POSITIVE).ln();
        }
    }
    (selected, selected_probability.ln(), entropy)
}

#[inline]
fn move_delta(action: u8) -> Option<(i16, i16)> {
    match action {
        1 => Some((0, -1)),
        2 => Some((0, 1)),
        3 => Some((1, 0)),
        4 => Some((-1, 0)),
        _ => None,
    }
}

#[inline]
fn pickup_spec(action: u8) -> Option<(usize, u16)> {
    match action {
        6..=21 => Some((0, u16::from(action - 5))),
        22..=29 => Some((8, u16::from(action - 21))),
        30..=33 => Some((9, u16::from(action - 29))),
        34..=37 => Some((10, u16::from(action - 33))),
        38..=41 => Some((11, u16::from(action - 37))),
        _ => None,
    }
}

#[inline]
fn place_animal(action: u8) -> Option<usize> {
    (42..=44)
        .contains(&action)
        .then(|| usize::from(action - 42))
}

#[inline]
fn unit_plant_crop(action: u8) -> Option<usize> {
    (45..=49)
        .contains(&action)
        .then(|| usize::from(action - 45))
}

fn remove_inventory_order(order: &mut [u8; PRIVATE_ITEMS], item: usize) {
    if let Some(index) = order.iter().position(|&entry| usize::from(entry) == item) {
        order.copy_within(index + 1.., index);
        order[PRIVATE_ITEMS - 1] = u8::MAX;
    }
}

#[inline]
fn animal_structure(animal: usize) -> TileKind {
    if animal == 0 {
        TileKind::Coop
    } else {
        TileKind::Pasture
    }
}

#[inline]
fn is_shed_access(x: usize, y: usize) -> bool {
    matches!((x, y), (4, 4) | (5, 4) | (4, 5) | (5, 5))
}

#[inline]
fn default_spawn() -> Position {
    Position(4, 4)
}

fn spawn_hand(farm: &Farm) -> Position {
    const ACCESS: [Position; 4] = [
        Position(4, 4),
        Position(5, 4),
        Position(4, 5),
        Position(5, 5),
    ];
    let mut occupancy = [0u8; 4];
    for position in &farm.positions[..usize::from(farm.units)] {
        if let Some(index) = ACCESS.iter().position(|candidate| candidate == position) {
            occupancy[index] += 1;
        }
    }
    let index = (0..4)
        .min_by_key(|&index| (occupancy[index], index))
        .unwrap();
    ACCESS[index]
}

#[inline]
fn quadrant_of(x: usize, y: usize) -> usize {
    match (y < 5, x < 5) {
        (true, true) => 0,
        (true, false) => 1,
        (false, true) => 2,
        (false, false) => 3,
    }
}

fn fib(n: u8) -> i64 {
    let (mut a, mut b) = (1i64, 1i64);
    for _ in 0..n {
        (a, b) = (b, a + b);
    }
    a
}

#[derive(Clone, Copy)]
enum Shape {
    Linear,
    Square,
    Sqrt,
    Log,
}

const MARKET_PARAMS: [(f64, f64, Shape, f64, Shape, f64); PRODUCTS] = [
    (25.0, 400.0, Shape::Sqrt, 0.80, Shape::Log, 0.20),
    (35.0, 450.0, Shape::Log, 0.20, Shape::Sqrt, 0.70),
    (60.0, 200.0, Shape::Linear, 0.40, Shape::Sqrt, 0.60),
    (120.0, 100.0, Shape::Sqrt, 0.70, Shape::Linear, 1.60),
    (250.0, 300.0, Shape::Log, 0.20, Shape::Square, 3.60),
    (50.0, 332.0, Shape::Linear, 0.40, Shape::Log, 0.20),
    (160.0, 122.0, Shape::Sqrt, 0.60, Shape::Linear, 1.60),
    (200.0, 105.0, Shape::Log, 0.20, Shape::Square, 3.20),
    (100.0, 200.0, Shape::Linear, 0.40, Shape::Linear, 0.40),
];

fn money_feature(amount: i64) -> f32 {
    let value = amount as f64;
    (value.signum() * value.abs().ln_1p() / 12.0) as f32
}

fn private_vector(private: &PrivateState, output: &mut [f32; 29]) {
    #[allow(clippy::needless_range_loop)]
    for item in 0..PRIVATE_ITEMS {
        output[item] = f32::from(private.shed[item]) / 100.0;
    }
    for crop in 0..CROPS {
        output[PRIVATE_ITEMS + crop] = f32::from(private.seeds[crop]) / 100.0;
    }
    for item in 0..PRIVATE_ITEMS {
        let aggregate: u16 = private
            .inventories
            .iter()
            .map(|inventory| inventory[item])
            .sum();
        output[PRIVATE_ITEMS + CROPS + item] = f32::from(aggregate) / 100.0;
    }
}

fn encode_farm(farm: &Farm, day: u16, step: u16, output: &mut [f32]) {
    debug_assert_eq!(output.len(), FARM_CHANNELS * TILE_COUNT);
    for (tile_index, tile) in farm.tiles.iter().copied().enumerate() {
        let mut set =
            |channel: usize, value: f32| output[channel * TILE_COUNT + tile_index] = value;
        if tile.kind == TileKind::Locked {
            set(0, 1.0);
            continue;
        }
        set(28, 1.0);
        match tile.kind {
            TileKind::Empty => set(1, 1.0),
            TileKind::Weed => set(2, 1.0),
            TileKind::Plant => {
                let crop = usize::from(tile.species);
                set(3 + crop, 1.0);
                set(13, f32::from(tile.yield_units) / 6.0);
                set(14, f32::from(day.saturating_sub(tile.origin_day)) / 30.0);
                set(15, f32::from(u8::from(tile.watered_or_fed)));
                set(16, (f32::from(tile.consecutive_unmet) / 2.0).min(1.0));
                set(
                    17,
                    ((f32::from(tile.fertilized_until_day) - f32::from(day) + 1.0) / 3.0).max(0.0),
                );
                if tile.max_lifespan_step >= 0 {
                    set(
                        25,
                        ((f32::from(tile.max_lifespan_step) - f32::from(step)) / 96.0)
                            .clamp(0.0, 1.0),
                    );
                    set(
                        26,
                        f32::from(u8::from(tile.max_lifespan_step <= step as i16)),
                    );
                    set(
                        27,
                        f32::from(u8::from(
                            tile.max_lifespan_step <= step as i16
                                && (step as i16 - tile.max_lifespan_step) % 2 == 0,
                        )),
                    );
                }
            }
            TileKind::Coop | TileKind::Pasture => {
                set(if tile.kind == TileKind::Coop { 8 } else { 9 }, 1.0);
                if tile.has_animal {
                    set(10 + usize::from(tile.species), 1.0);
                    set(13, f32::from(tile.yield_units) / 6.0);
                    set(14, f32::from(day.saturating_sub(tile.origin_day)) / 30.0);
                    set(18, f32::from(u8::from(tile.watered_or_fed)));
                    set(19, (f32::from(tile.consecutive_unmet) / 2.0).min(1.0));
                    set(20, f32::from(u8::from(tile.cared_today)));
                    set(21, f32::from(u8::from(tile.fertilizer_available)));
                    set(22, (f32::from(tile.pending_care_bonus) / 5.0).min(1.0));
                }
            }
            TileKind::Locked => unreachable!(),
        }
    }
    let main = farm.positions[0];
    output[23 * TILE_COUNT + usize::from(main.1) * BOARD_SIZE + usize::from(main.0)] = 1.0;
    for position in &farm.positions[1..usize::from(farm.units)] {
        output[24 * TILE_COUNT + usize::from(position.1) * BOARD_SIZE + usize::from(position.0)] +=
            1.0 / MAX_UNITS as f32;
    }
}

// Structured tile continuous field indices, matching tokens.py
// TILE_CONTINUOUS_FIELDS order exactly.
const TF_YIELD_FRACTION: usize = 0;
const TF_AGE_FRACTION: usize = 1;
const TF_MATURITY_FRACTION: usize = 2;
const TF_WATERED_TODAY: usize = 3;
const TF_FED_TODAY: usize = 4;
const TF_CARED_TODAY: usize = 5;
const TF_FERTILIZER_REMAINING: usize = 6;
const TF_FERTILIZER_AVAILABLE: usize = 7;
const TF_PENDING_CARE_BONUS: usize = 8;
const TF_DECAY_PRESSURE: usize = 9;
const TF_LIFESPAN_REMAINING: usize = 10;
const TF_LIFESPAN_EXPIRED: usize = 11;
const TF_LIFESPAN_DECAY_TICK: usize = 12;
const TF_HARVEST_READY: usize = 13;
const TF_EDGE: usize = 14;
const TF_CORNER: usize = 15;
const TF_SHED_DISTANCE: usize = 16;
const TF_SHED_ACCESS: usize = 17;

/// Structured tile kind vocabulary index (tokens.py TILE_KINDS order).
#[inline]
fn structured_tile_kind(kind: TileKind) -> i8 {
    match kind {
        TileKind::Locked => 0,
        TileKind::Empty => 1,
        TileKind::Weed => 2,
        TileKind::Plant => 3,
        TileKind::Coop => 4,
        TileKind::Pasture => 5,
    }
}

fn encode_farm_structured(
    farm: &Farm,
    day: u16,
    step: u16,
    opponent: bool,
    categorical: &mut [i8],
    continuous: &mut [f32],
) {
    debug_assert_eq!(categorical.len(), TILE_COUNT * TILE_CATEGORICAL);
    debug_assert_eq!(continuous.len(), TILE_COUNT * TILE_CONTINUOUS);
    const ACCESS: [(i16, i16); 4] = [(4, 4), (5, 4), (4, 5), (5, 5)];
    // The 10x10 board's largest Manhattan distance to shed access is 8.
    const MAX_SHED_DISTANCE: f64 = 8.0;
    for y in 0..BOARD_SIZE {
        for x in 0..BOARD_SIZE {
            let token = y * BOARD_SIZE + x;
            let tile = farm.tiles[token];
            let row = &mut categorical[token * TILE_CATEGORICAL..(token + 1) * TILE_CATEGORICAL];
            row[0] = structured_tile_kind(tile.kind);
            row[2] = i8::from(opponent);
            row[3] = y as i8;
            row[4] = x as i8;
            row[5] = quadrant_of(x, y) as i8;
            let features = &mut continuous[token * TILE_CONTINUOUS..(token + 1) * TILE_CONTINUOUS];
            let edge = x == 0 || x == BOARD_SIZE - 1 || y == 0 || y == BOARD_SIZE - 1;
            let corner = (x == 0 || x == BOARD_SIZE - 1) && (y == 0 || y == BOARD_SIZE - 1);
            features[TF_EDGE] = f32::from(u8::from(edge));
            features[TF_CORNER] = f32::from(u8::from(corner));
            let distance = ACCESS
                .iter()
                .map(|&(ax, ay)| (x as i16 - ax).abs() + (y as i16 - ay).abs())
                .min()
                .unwrap();
            features[TF_SHED_DISTANCE] = (f64::from(distance) / MAX_SHED_DISTANCE) as f32;
            features[TF_SHED_ACCESS] = f32::from(u8::from(is_shed_access(x, y)));

            let age = f64::from(day.saturating_sub(tile.origin_day));
            let episode_days = 30.0;
            match tile.kind {
                TileKind::Locked | TileKind::Empty | TileKind::Weed => {}
                TileKind::Plant => {
                    let crop = usize::from(tile.species);
                    row[1] = (1 + crop) as i8;
                    let stock = f64::from(tile.yield_units);
                    features[TF_YIELD_FRACTION] =
                        (stock / f64::from(CROP_MAX_HELD[crop])).min(1.0) as f32;
                    features[TF_AGE_FRACTION] = (age / episode_days).min(1.0) as f32;
                    features[TF_MATURITY_FRACTION] =
                        (age / f64::from(FIRST_YIELD[crop])).min(1.0) as f32;
                    features[TF_WATERED_TODAY] = f32::from(u8::from(tile.watered_or_fed));
                    features[TF_FERTILIZER_REMAINING] = (f64::from(
                        (i32::from(tile.fertilized_until_day) - i32::from(day) + 1).max(0),
                    ) / 3.0)
                        .min(1.0) as f32;
                    features[TF_DECAY_PRESSURE] =
                        (f64::from(tile.consecutive_unmet) / 2.0).min(1.0) as f32;
                    if tile.max_lifespan_step >= 0 {
                        features[TF_LIFESPAN_REMAINING] = (f64::from(
                            (i32::from(tile.max_lifespan_step) - i32::from(step)).max(0),
                        ) / 96.0)
                            .min(1.0)
                            as f32;
                        let expired = i32::from(step) >= i32::from(tile.max_lifespan_step);
                        features[TF_LIFESPAN_EXPIRED] = f32::from(u8::from(expired));
                        features[TF_LIFESPAN_DECAY_TICK] = f32::from(u8::from(
                            expired
                                && (i32::from(step) - i32::from(tile.max_lifespan_step)) % 2 == 0,
                        ));
                    }
                    features[TF_HARVEST_READY] = f32::from(u8::from(
                        age >= f64::from(FIRST_YIELD[crop]) && tile.yield_units > 0,
                    ));
                }
                TileKind::Coop | TileKind::Pasture => {
                    if !tile.has_animal {
                        continue;
                    }
                    let animal = usize::from(tile.species);
                    row[1] = (1 + CROPS + animal) as i8;
                    let stock = f64::from(tile.yield_units);
                    features[TF_YIELD_FRACTION] =
                        (stock / f64::from(ANIMAL_MAX_HELD[animal])).min(1.0) as f32;
                    features[TF_AGE_FRACTION] = (age / episode_days).min(1.0) as f32;
                    features[TF_MATURITY_FRACTION] =
                        (age / f64::from(ANIMAL_FIRST_YIELD[animal])).min(1.0) as f32;
                    features[TF_FED_TODAY] = f32::from(u8::from(tile.watered_or_fed));
                    features[TF_CARED_TODAY] = f32::from(u8::from(tile.cared_today));
                    features[TF_FERTILIZER_AVAILABLE] =
                        f32::from(u8::from(tile.fertilizer_available));
                    features[TF_PENDING_CARE_BONUS] =
                        (f64::from(tile.pending_care_bonus) / 5.0).min(1.0) as f32;
                    features[TF_DECAY_PRESSURE] =
                        (f64::from(tile.consecutive_unmet) / 2.0).min(1.0) as f32;
                    features[TF_HARVEST_READY] = f32::from(u8::from(tile.yield_units > 0));
                }
            }
        }
    }
}

#[inline]
fn shape(kind: Shape, x: f64) -> f64 {
    match kind {
        Shape::Linear => x,
        Shape::Square => x * x,
        Shape::Sqrt => x.sqrt(),
        Shape::Log => x.ln_1p(),
    }
}

fn relative_score(zero: f64, one: f64) -> f32 {
    let total = zero + one;
    if total == 0.0 {
        0.0
    } else {
        ((zero - one) / total) as f32
    }
}

pub fn market_price(item: usize, inventory: i32) -> i64 {
    let (base, scale, below_shape, below_target, above_shape, above_target) = MARKET_PARAMS[item];
    let (kind, target, distance, sign) = if inventory < MARKET_I0 {
        (
            below_shape,
            below_target,
            f64::from(MARKET_I0 - inventory),
            1.0,
        )
    } else {
        (
            above_shape,
            above_target,
            f64::from(inventory - MARKET_I0),
            -1.0,
        )
    };
    let amplitude = target * base / shape(kind, scale);
    round_ties_even(base + sign * amplitude * shape(kind, distance)).max(PRICE_FLOOR)
}

#[inline]
fn round_ties_even(value: f64) -> i64 {
    let floor = value.floor();
    let fraction = value - floor;
    if fraction < 0.5 {
        floor as i64
    } else if fraction > 0.5 {
        floor as i64 + 1
    } else {
        let integer = floor as i64;
        if integer % 2 == 0 {
            integer
        } else {
            integer + 1
        }
    }
}

#[derive(Serialize)]
struct Snapshot<'a> {
    step: u16,
    day: u16,
    hour: u16,
    done: bool,
    farms: Vec<FarmSnapshot>,
    privates: Vec<PrivateSnapshot>,
    market: MarketSnapshot,
    town: TownSnapshot,
    rewards: [f64; PLAYERS],
    statuses: [&'static str; PLAYERS],
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    #[serde(skip)]
    _marker: std::marker::PhantomData<&'a ()>,
}

#[derive(Serialize)]
struct FarmSnapshot {
    money: f64,
    tiles: Vec<Vec<serde_json::Value>>,
    farmer: Position,
    hands: Vec<Position>,
    unlocked_quadrants: Vec<&'static str>,
    hires_today: u8,
}

#[derive(Serialize)]
struct PrivateSnapshot {
    shed: serde_json::Map<String, serde_json::Value>,
    seeds: serde_json::Map<String, serde_json::Value>,
    inventories: Vec<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Serialize)]
struct MarketSnapshot {
    inventory: serde_json::Map<String, serde_json::Value>,
    prices: serde_json::Map<String, serde_json::Value>,
}

#[derive(Serialize)]
struct TownSnapshot {
    unlocked_shops: Vec<&'static str>,
}

impl Game {
    pub fn snapshot_json(&self, include_seed: bool) -> String {
        let farms = self.farms.iter().map(farm_snapshot).collect();
        let privates = self
            .privates
            .iter()
            .zip(self.farms.iter())
            .map(|(private, farm)| private_snapshot(private, usize::from(farm.units)))
            .collect();
        let inventory = PRODUCT_NAMES
            .iter()
            .enumerate()
            .map(|(index, name)| ((*name).to_owned(), self.market_inventory[index].into()))
            .collect();
        let prices = PRODUCT_NAMES
            .iter()
            .enumerate()
            .map(|(index, name)| ((*name).to_owned(), self.market_prices[index].into()))
            .collect();
        let terminal_rewards = if self.done {
            [self.farms[0].money as f64, self.farms[1].money as f64]
        } else {
            [0.0, 0.0]
        };
        serde_json::to_string(&Snapshot {
            step: self.step,
            day: self.step / self.config.turns_per_day,
            hour: self.step % self.config.turns_per_day,
            done: self.done,
            farms,
            privates,
            market: MarketSnapshot { inventory, prices },
            town: TownSnapshot {
                unlocked_shops: self.shops[..usize::from(self.shop_count)]
                    .iter()
                    .map(|&index| SHOP_NAMES_SORTED[usize::from(index)])
                    .collect(),
            },
            rewards: terminal_rewards,
            statuses: if self.done {
                ["DONE", "DONE"]
            } else {
                ["ACTIVE", "ACTIVE"]
            },
            seed: include_seed.then_some(self.seed),
            _marker: std::marker::PhantomData,
        })
        .expect("serializing an in-memory game cannot fail")
    }
}

fn farm_snapshot(farm: &Farm) -> FarmSnapshot {
    let tiles = (0..BOARD_SIZE)
        .map(|y| {
            (0..BOARD_SIZE)
                .map(|x| tile_json(farm.tiles[y * BOARD_SIZE + x]))
                .collect()
        })
        .collect();
    const NAMES: [&str; 4] = ["NW", "NE", "SW", "SE"];
    FarmSnapshot {
        money: farm.money as f64,
        tiles,
        farmer: farm.positions[0],
        hands: farm.positions[1..usize::from(farm.units)].to_vec(),
        unlocked_quadrants: (0..4)
            .filter(|&index| farm.unlocked & (1 << index) != 0)
            .map(|index| NAMES[index])
            .collect(),
        hires_today: farm.hires_today,
    }
}

fn tile_json(tile: Tile) -> serde_json::Value {
    use serde_json::{Value, json};
    match tile.kind {
        TileKind::Empty => Value::Null,
        TileKind::Locked => Value::String("LOCKED".to_owned()),
        TileKind::Weed => json!({"kind": "WEED"}),
        TileKind::Plant => json!({
            "kind": "PLANT",
            "crop": CROP_NAMES[usize::from(tile.species)],
            "planted_day": tile.origin_day,
            "watered_today": tile.watered_or_fed,
            "consecutive_unwatered": tile.consecutive_unmet,
            "yield_units": tile.yield_units,
            "max_lifespan_step": tile.max_lifespan_step,
            "fertilized_until_day": tile.fertilized_until_day,
        }),
        TileKind::Coop | TileKind::Pasture if !tile.has_animal => {
            json!({"kind": if tile.kind == TileKind::Coop { "COOP" } else { "PASTURE" }})
        }
        TileKind::Coop | TileKind::Pasture => json!({
            "kind": if tile.kind == TileKind::Coop { "COOP" } else { "PASTURE" },
            "animal": ANIMAL_NAMES[usize::from(tile.species)],
            "placed_day": tile.origin_day,
            "yield_units": tile.yield_units,
            "consecutive_unfed": tile.consecutive_unmet,
            "fed_today": tile.watered_or_fed,
            "cared_today": tile.cared_today,
            "fertilizer_available": tile.fertilizer_available,
            "pending_care_bonus": tile.pending_care_bonus,
        }),
    }
}

fn private_snapshot(private: &PrivateState, units: usize) -> PrivateSnapshot {
    let shed = PRIVATE_NAMES
        .iter()
        .enumerate()
        .map(|(index, name)| ((*name).to_owned(), private.shed[index].into()))
        .collect();
    let seeds = CROP_NAMES
        .iter()
        .enumerate()
        .map(|(index, name)| ((*name).to_owned(), private.seeds[index].into()))
        .collect();
    let inventories = (0..units)
        .map(|unit| {
            private.inventory_order[unit]
                .iter()
                .take_while(|&&item| item != u8::MAX)
                .map(|&raw_item| {
                    let item = usize::from(raw_item);
                    (
                        PRIVATE_NAMES[item].to_owned(),
                        private.inventories[unit][item].into(),
                    )
                })
                .collect()
        })
        .collect();
    PrivateSnapshot {
        shed,
        seeds,
        inventories,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_and_terminal_horizon() {
        let mut game = Game::new(4, GameConfig::default());
        assert_eq!(game.step, 0);
        assert_eq!(
            game.farms[0].tiles[4 * BOARD_SIZE + 4].kind,
            TileKind::Empty
        );
        let pass = [CompactAction::default(); PLAYERS];
        for _ in 0..718 {
            assert!(!game.step(&pass).done);
        }
        assert!(game.step(&pass).done);
        assert_eq!(game.step, 719);
    }

    #[test]
    fn price_reference_points() {
        assert_eq!(market_price(0, 10_000), 25);
        assert_eq!(market_price(0, 9_600), 45);
        assert_eq!(market_price(4, 10_300), 1);
    }

    #[test]
    fn pair_potential_is_relative_farm_value_with_exact_liquid_core() {
        let mut game = Game::new(0, GameConfig::default());
        game.farms[0].money = 3000;
        game.farms[1].money = 1000;
        assert_eq!(game.pair_potential(), 0.5);

        game.farms[0].money = 1000;
        game.farms[1].money = 3000;
        assert_eq!(game.pair_potential(), -0.5);

        game.farms[0].money = 0;
        game.farms[1].money = 0;
        assert_eq!(game.pair_potential(), 0.0);

        // Held products count at their exact sale proceeds, whether they sit
        // in the shed or in a hired unit's hands.
        game.farms[0].money = 1000;
        game.farms[1].money = 1000;
        game.farms[0].units = 2;
        game.privates[0].shed[0] = 30;
        game.privates[0].inventories[1][0] = 10;
        game.privates[1].shed[4] = 5;
        let zero = game.liquidation_value(0);
        let one = game.liquidation_value(1);
        assert!(zero > 1000.0);
        assert!(one > 1000.0);
        assert_eq!(game.pair_potential(), ((zero - one) / (zero + one)) as f32);

        // Unhired unit slots are outside the observation and must not count.
        game.privates[0].inventories[5][0] = 99;
        assert_eq!(game.liquidation_value(0), zero);
    }

    #[test]
    fn liquidation_value_matches_engine_sell_proceeds_exactly() {
        let mut game = Game::new(0, GameConfig::default());
        game.farms[0].money = 250;
        game.privates[0].shed[0] = 60;
        game.privates[0].shed[4] = 25;
        game.privates[0].shed[8] = 40;
        // Deep above-target inventory drives MELON quotes onto the price
        // floor, exercising the floor-conditional market restock.
        game.market_inventory[4] = 10_320;
        let predicted = game.liquidation_value(0);

        for item in [0, 4, 8] {
            while game.privates[0].shed[item] > 0 {
                let price = market_price(item, game.market_inventory[item]);
                let order = MarketOrder {
                    kind: 13 + item as u8,
                    item,
                    remaining: 1,
                };
                assert!(game.commit_market_unit(0, order, price));
            }
        }

        assert_eq!(predicted, game.farms[0].money as f64);
    }

    #[test]
    fn illiquid_value_credits_cost_basis_fractions() {
        let mut game = Game::new(0, GameConfig::default());
        assert_eq!(game.illiquid_value(0), 0.0);
        assert_eq!(game.illiquid_value(1), 0.0);

        game.privates[0].shed[9] = 2; // geese in the shed
        game.privates[0].seeds[2] = 4; // tomato seeds
        game.farms[0].tiles[3].kind = TileKind::Plant;
        game.farms[0].tiles[3].species = 0; // wheat
        game.farms[0].tiles[3].yield_units = 2;
        // All three species, pinning the species -> product price mapping
        // (goose -> egg, cow -> milk, sheep -> wool).
        game.farms[0].tiles[7].has_animal = true;
        game.farms[0].tiles[7].species = 1; // cow
        game.farms[0].tiles[7].yield_units = 3;
        game.farms[0].tiles[9].has_animal = true;
        game.farms[0].tiles[9].species = 0; // goose
        game.farms[0].tiles[9].yield_units = 1;
        game.farms[0].tiles[11].has_animal = true;
        game.farms[0].tiles[11].species = 2; // sheep
        game.farms[0].tiles[11].yield_units = 2;
        game.farms[0].unlocked = 0b111;

        let expected = ILLIQUID_SHED_ANIMAL_CREDIT * 2.0 * ANIMAL_COST[0] as f64
            + ILLIQUID_SHED_SEED_CREDIT * 4.0 * SEED_COST[2] as f64
            + ILLIQUID_PLANTED_SEED_CREDIT * SEED_COST[0] as f64
            + ILLIQUID_PENDING_YIELD_CREDIT * 2.0 * game.market_prices[0] as f64
            + ILLIQUID_PLACED_ANIMAL_CREDIT * ANIMAL_COST[1] as f64
            + ILLIQUID_PENDING_YIELD_CREDIT * 3.0 * game.market_prices[6] as f64 // cow -> MILK
            + ILLIQUID_PLACED_ANIMAL_CREDIT * ANIMAL_COST[0] as f64
            + ILLIQUID_PENDING_YIELD_CREDIT * 1.0 * game.market_prices[5] as f64 // goose -> EGG
            + ILLIQUID_PLACED_ANIMAL_CREDIT * ANIMAL_COST[2] as f64
            + ILLIQUID_PENDING_YIELD_CREDIT * 2.0 * game.market_prices[7] as f64 // sheep -> WOOL
            + ILLIQUID_LAND_CREDIT * (LAND_PRICES[0] + LAND_PRICES[1]) as f64;
        assert!((game.illiquid_value(0) - expected).abs() < 1e-9);
        // Investment now moves the shaping potential instead of reading as
        // pure loss, so self-play cannot settle into the never-spend tie.
        assert!(game.pair_potential() > 0.0);
    }

    #[test]
    fn terminal_potential_scores_bank_only() {
        let mut game = Game::new(0, GameConfig::default());
        game.farms[0].money = 3000;
        game.farms[1].money = 1000;
        game.privates[1].shed.fill(100);
        assert_eq!(game.terminal_potential(), 0.5);
        assert!(game.pair_potential() < 0.5);

        assert_eq!(game.post_step_potential(), game.pair_potential());
        game.done = true;
        assert_eq!(game.post_step_potential(), 0.5);
    }

    #[test]
    fn exact_pickup_quantities_and_sequential_reservation() {
        let mut game = Game::new(0, GameConfig::default());
        game.privates[0].shed[0] = 16;
        game.privates[0].shed[8] = 8;
        game.privates[0].shed[9] = 4;
        let mut action = CompactAction::default();
        action.market_kinds[0] = 1;
        action.market_kinds[1] = 1;
        action.units[0] = 21; // wheat 16
        game.step(&[action, CompactAction::default()]);
        assert_eq!(game.privates[0].inventories[0][0], 16);
        assert_eq!(game.farms[0].units, 3);
        let mut action = CompactAction::default();
        action.units[0] = 29; // fertilizer 8
        action.units[1] = 33; // goose 4
        game.step(&[action, CompactAction::default()]);
        assert_eq!(game.privates[0].inventories[0][8], 8);
        assert_eq!(game.privates[0].inventories[1][9], 4);
    }

    #[test]
    fn simultaneous_market_units_share_quote_then_commit_player_order() {
        let mut game = Game::new(0, GameConfig::default());
        game.privates[0].shed[0] = 2;
        game.privates[1].shed[0] = 2;
        let mut sell = CompactAction::default();
        sell.market_kinds[0] = 13;
        sell.market_quantities[0] = 1; // exact quantity two
        game.step(&[sell, sell]);
        // Both first units see $25; both second units see the same post-pair quote.
        let second_quote = market_price(0, MARKET_I0 + 2);
        assert_eq!(game.farms[0].money, 3000 + 25 + second_quote);
        assert_eq!(game.farms[1].money, 3000 + 25 + second_quote);
        // Town-center demand at step zero consumes one after the four sales.
        assert_eq!(game.market_inventory[0], MARKET_I0 + 3);
    }

    #[test]
    fn plant_water_harvest_and_daily_weed_lifecycle() {
        let mut game = Game::new(0, GameConfig::default());
        game.privates[0].seeds[0] = 1;
        let mut plant = CompactAction::default();
        plant.units[0] = 45;
        game.step(&[plant, CompactAction::default()]);
        assert_eq!(game.farms[0].tiles[44].kind, TileKind::Plant);
        // Unwatered planting day + one refresh makes two misses and a weed.
        for _ in 1..24 {
            game.step(&[CompactAction::default(); PLAYERS]);
        }
        assert_eq!(game.farms[0].tiles[44].kind, TileKind::Weed);
        let mut dig = CompactAction::default();
        dig.units[0] = 53;
        game.step(&[dig, CompactAction::default()]);
        assert_eq!(game.farms[0].tiles[44].kind, TileKind::Empty);
    }

    #[test]
    fn end_of_day_rng_shop_sequence_is_stable() {
        let mut game = Game::new(123, GameConfig::default());
        for _ in 0..72 {
            game.step(&[CompactAction::default(); PLAYERS]);
        }
        assert_eq!(game.shop_count, 1);
        // Differentially established against random.Random((123*1_000_003)^2)
        // after the exact two-farm empty-tile weed draw sequence.
        assert_eq!(SHOP_NAMES_SORTED[usize::from(game.shops[0])], "BAKERY");
        let weeds_zero: Vec<usize> = game.farms[0]
            .tiles
            .iter()
            .enumerate()
            .filter(|(_, tile)| tile.kind == TileKind::Weed)
            .map(|(index, _)| index)
            .collect();
        assert_eq!(weeds_zero, [4]);
        assert!(
            game.farms[1]
                .tiles
                .iter()
                .all(|tile| tile.kind != TileKind::Weed)
        );
        assert_eq!(game.step, 72);
    }

    #[test]
    fn quantity_mask_is_exact_prefix_not_sparse_bins() {
        let mut game = Game::new(0, GameConfig::default());
        game.privates[0].shed[0] = 37;
        let mut action = CompactAction::default();
        action.market_kinds[0] = 13;
        let masks = game.factor_masks(0, &action);
        let row = &masks.market_quantity[..MARKET_QUANTITIES];
        assert!(row[..37].iter().all(|&valid| valid));
        assert!(row[37..].iter().all(|&valid| !valid));
    }

    #[test]
    fn place_animal_at_shed_reserves_capacity_before_later_units() {
        let mut game = Game::new(0, GameConfig::default());
        game.farms[0].units = 2;
        game.farms[0].positions[0] = Position(5, 4);
        game.farms[0].positions[1] = Position(4, 4);
        game.privates[0].shed[0] = 99;
        game.add_inventory(0, 0, 10, 1);
        game.add_inventory(0, 1, 10, 1);
        let mut action = CompactAction::default();
        action.units[0] = 43;
        action.units[1] = 43;

        let masks = game.factor_masks(0, &action);

        assert!(masks.unit[43]);
        assert!(!masks.unit[UNIT_ACTIONS + 43]);
        game.step(&[action, CompactAction::default()]);
        assert_eq!(
            game.farms[0].tiles[4 * BOARD_SIZE + 5].kind,
            TileKind::Locked
        );
        assert_eq!(game.privates[0].shed[10], 1);
        assert_eq!(game.privates[0].inventories[0][10], 0);
        assert_eq!(game.privates[0].inventories[1][10], 1);
    }

    #[test]
    fn place_animal_prefers_matching_structure_when_shed_is_full() {
        let mut game = Game::new(0, GameConfig::default());
        game.farms[0].tiles[44] = Tile::structure(TileKind::Pasture);
        game.privates[0].shed[0] = 100;
        game.add_inventory(0, 0, 10, 1);
        let mut action = CompactAction::default();
        action.units[0] = 43;

        let masks = game.factor_masks(0, &action);

        assert!(masks.unit[43]);
        game.step(&[action, CompactAction::default()]);
        assert!(game.farms[0].tiles[44].has_animal);
        assert_eq!(game.farms[0].tiles[44].species, 1);
        assert_eq!(game.privates[0].shed.iter().sum::<u16>(), 100);
        assert_eq!(game.privates[0].inventories[0][10], 0);
    }

    #[test]
    fn build_then_place_does_not_consume_shed_capacity_in_unit_ledger() {
        let mut game = Game::new(0, GameConfig::default());
        game.farms[0].units = 2;
        game.privates[0].shed[0] = 99;
        game.add_inventory(0, 1, 10, 1);
        let mut action = CompactAction::default();
        action.units[0] = 55;
        action.units[1] = 43;

        let masks = game.factor_masks(0, &action);

        assert!(masks.unit[UNIT_ACTIONS + 43]);
        assert!(masks.market_kind[8]);
        game.step(&[action, CompactAction::default()]);
        assert!(game.farms[0].tiles[44].has_animal);
        assert_eq!(game.privates[0].shed.iter().sum::<u16>(), 99);
    }

    #[test]
    fn inventory_overflow_preserves_python_dict_insertion_order() {
        let config = GameConfig {
            shed_capacity: 1,
            ..GameConfig::default()
        };
        let mut game = Game::new(0, config);
        game.add_inventory(0, 0, 11, 1);
        game.add_inventory(0, 0, 0, 1);
        game.drop_inventory(0, 0);
        assert_eq!(game.privates[0].shed[11], 1);
        assert_eq!(game.privates[0].shed[0], 0);
        assert_eq!(
            game.privates[0].inventory_order[0],
            [u8::MAX; PRIVATE_ITEMS]
        );
    }

    #[test]
    fn animal_care_feed_production_and_escape() {
        let mut game = Game::new(0, GameConfig::default());
        game.farms[0].tiles[44] = Tile::animal(0, 0);
        game.farms[0].tiles[44].watered_or_fed = true;
        game.farms[0].tiles[44].cared_today = true;
        game.daily_refresh_animals(0, 3);
        let goose = game.farms[0].tiles[44];
        assert_eq!(goose.yield_units, 1);
        assert_eq!(goose.pending_care_bonus, 1);
        assert!(goose.fertilizer_available);
        game.farms[0].tiles[44].watered_or_fed = true;
        game.daily_refresh_animals(0, 4);
        assert_eq!(game.farms[0].tiles[44].yield_units, 3);
        game.daily_refresh_animals(0, 5);
        assert!(game.farms[0].tiles[44].has_animal);
        game.daily_refresh_animals(0, 6);
        assert!(!game.farms[0].tiles[44].has_animal);
        assert_eq!(game.farms[0].tiles[44].kind, TileKind::Coop);
    }

    #[test]
    fn hire_then_land_orders_apply_in_slot_order() {
        let mut game = Game::new(0, GameConfig::default());
        let mut action = CompactAction::default();
        action.market_kinds[0] = 1;
        action.market_kinds[1] = 2;
        game.step(&[action, CompactAction::default()]);
        assert_eq!(game.farms[0].money, 1999);
        assert_eq!(game.farms[0].units, 2);
        assert_eq!(game.farms[0].unlocked, 0b0011);
        assert_eq!(game.farms[0].tiles[5].kind, TileKind::Empty);
    }

    #[test]
    fn bankers_rounding_matches_python_ties() {
        assert_eq!(round_ties_even(2.5), 2);
        assert_eq!(round_ties_even(3.5), 4);
        assert_eq!(round_ties_even(4.499_999), 4);
        assert_eq!(round_ties_even(4.500_001), 5);
    }

    #[test]
    fn categorical_zero_draw_never_selects_a_masked_prefix() {
        let logits = [100.0, 0.0, 1.0];
        let mask = [false, true, true];
        assert_eq!(sample_categorical(&logits, &mask, false, 1.0, 0.0).0, 1);
    }

    #[test]
    fn categorical_roundoff_fallback_is_last_valid_category() {
        let logits = [0.0, 0.0, 100.0];
        let mask = [true, true, false];
        assert_eq!(
            sample_categorical(&logits, &mask, false, 1.0, 0.999_999_94).0,
            1
        );
    }
}
