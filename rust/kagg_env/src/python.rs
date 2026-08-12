use crate::core::{
    BOARD_CHANNELS, BOARD_SIZE, CRITIC_FEATURES, CompactAction, GLOBAL_FEATURES, Game, GameConfig,
    MARKET_KINDS, MARKET_QUANTITIES, MAX_MARKET_ORDERS, MAX_UNITS, PLAYERS, UNIT_ACTIONS,
    UNIT_FEATURES,
};
use half::f16;
use numpy::ndarray::{Array1, Array2, Array3, Array4};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

#[pyclass(name = "BatchEnv", unsendable)]
pub(crate) struct BatchEnv {
    games: Vec<Game>,
}

#[pymethods]
impl BatchEnv {
    #[new]
    #[pyo3(signature = (seeds))]
    fn new(seeds: PyReadonlyArray1<'_, u64>) -> PyResult<Self> {
        let seeds = seeds.as_slice()?;
        if seeds.is_empty() {
            return Err(PyValueError::new_err("seeds must be non-empty"));
        }
        Ok(Self {
            games: seeds
                .iter()
                .map(|&seed| Game::new(seed, GameConfig::default()))
                .collect(),
        })
    }

    fn __len__(&self) -> usize {
        self.games.len()
    }

    fn reset(&mut self, seeds: PyReadonlyArray1<'_, u64>) -> PyResult<()> {
        let seeds = seeds.as_slice()?;
        if seeds.len() != self.games.len() {
            return Err(PyValueError::new_err(format!(
                "reset seed count {} does not match batch {}",
                seeds.len(),
                self.games.len()
            )));
        }
        for (game, &seed) in self.games.iter_mut().zip(seeds) {
            *game = Game::new(seed, GameConfig::default());
        }
        Ok(())
    }

    #[pyo3(signature = (index, include_seed=false))]
    fn snapshot_json(&self, index: usize, include_seed: bool) -> PyResult<String> {
        self.games
            .get(index)
            .map(|game| game.snapshot_json(include_seed))
            .ok_or_else(|| PyIndexError::new_err(format!("game index {index} out of bounds")))
    }

    /// Encode both seats in game-major/player-minor order without Python objects.
    fn encoded<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let rows = py.detach(|| {
            self.games
                .par_iter()
                .flat_map_iter(|game| (0..PLAYERS).map(|player| encode_row(game, player)))
                .collect::<Vec<_>>()
        });
        let batch_rows = rows.len();
        let mut board = Vec::with_capacity(batch_rows * BOARD_CHANNELS * BOARD_SIZE * BOARD_SIZE);
        let mut globals = Vec::with_capacity(batch_rows * GLOBAL_FEATURES);
        let mut critic = Vec::with_capacity(batch_rows * CRITIC_FEATURES);
        let mut units = Vec::with_capacity(batch_rows * MAX_UNITS * UNIT_FEATURES);
        let mut positions = Vec::with_capacity(batch_rows * MAX_UNITS * 2);
        let mut active = Vec::with_capacity(batch_rows * MAX_UNITS);
        for row in rows {
            board.extend(row.board.into_iter().map(f16::from_f32));
            globals.extend(row.globals.into_iter().map(f16::from_f32));
            critic.extend(row.critic.into_iter().map(f16::from_f32));
            units.extend(row.units.into_iter().map(f16::from_f32));
            positions.extend(row.positions);
            active.extend(row.active);
        }
        let output = PyDict::new(py);
        output.set_item(
            "board",
            Array4::from_shape_vec((batch_rows, BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE), board)
                .expect("encoded board shape is internal")
                .into_pyarray(py),
        )?;
        output.set_item(
            "global_features",
            Array2::from_shape_vec((batch_rows, GLOBAL_FEATURES), globals)
                .expect("encoded globals shape is internal")
                .into_pyarray(py),
        )?;
        output.set_item(
            "critic_features",
            Array2::from_shape_vec((batch_rows, CRITIC_FEATURES), critic)
                .expect("encoded critic shape is internal")
                .into_pyarray(py),
        )?;
        output.set_item(
            "units",
            Array3::from_shape_vec((batch_rows, MAX_UNITS, UNIT_FEATURES), units)
                .expect("encoded units shape is internal")
                .into_pyarray(py),
        )?;
        output.set_item(
            "unit_positions",
            Array3::from_shape_vec((batch_rows, MAX_UNITS, 2), positions)
                .expect("encoded position shape is internal")
                .into_pyarray(py),
        )?;
        output.set_item(
            "unit_active",
            Array2::from_shape_vec((batch_rows, MAX_UNITS), active)
                .expect("encoded active shape is internal")
                .into_pyarray(py),
        )?;
        output.set_item(
            "potentials",
            Array1::from_vec(self.games.iter().map(Game::pair_potential).collect())
                .into_pyarray(py),
        )?;
        Ok(output)
    }

    fn step_factors<'py>(
        &mut self,
        py: Python<'py>,
        unit_actions: PyReadonlyArray3<'py, u8>,
        market_kinds: PyReadonlyArray3<'py, u8>,
        market_quantities: PyReadonlyArray3<'py, u8>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let expected_units = [self.games.len(), PLAYERS, MAX_UNITS];
        let expected_market = [self.games.len(), PLAYERS, MAX_MARKET_ORDERS];
        if unit_actions.shape() != expected_units {
            return Err(PyValueError::new_err(format!(
                "unit_actions shape {:?}, expected {expected_units:?}",
                unit_actions.shape()
            )));
        }
        if market_kinds.shape() != expected_market || market_quantities.shape() != expected_market {
            return Err(PyValueError::new_err(format!(
                "market factor shapes {:?}/{:?}, expected {expected_market:?}",
                market_kinds.shape(),
                market_quantities.shape()
            )));
        }
        let units = unit_actions.as_array();
        let kinds = market_kinds.as_array();
        let quantities = market_quantities.as_array();
        let compact: Vec<[CompactAction; PLAYERS]> = (0..self.games.len())
            .map(|game| {
                std::array::from_fn(|player| CompactAction {
                    units: std::array::from_fn(|unit| units[[game, player, unit]]),
                    market_kinds: std::array::from_fn(|slot| kinds[[game, player, slot]]),
                    market_quantities: std::array::from_fn(|slot| quantities[[game, player, slot]]),
                })
            })
            .collect();
        let results = py.detach(|| {
            self.games
                .par_iter_mut()
                .zip(compact.par_iter())
                .map(|(game, actions)| game.step(actions))
                .collect::<Vec<_>>()
        });
        let mut rewards = Vec::with_capacity(results.len() * PLAYERS);
        let mut money = Vec::with_capacity(results.len() * PLAYERS);
        let mut dones = Vec::with_capacity(results.len());
        for result in results {
            rewards.extend(result.rewards);
            money.extend(result.money);
            dones.push(result.done);
        }
        let output = PyDict::new(py);
        output.set_item(
            "rewards",
            Array2::from_shape_vec((self.games.len(), PLAYERS), rewards)
                .expect("step reward shape is internal")
                .into_pyarray(py),
        )?;
        output.set_item(
            "final_money",
            Array2::from_shape_vec((self.games.len(), PLAYERS), money)
                .expect("step money shape is internal")
                .into_pyarray(py),
        )?;
        output.set_item("dones", dones.into_pyarray(py))?;
        Ok(output)
    }
}

struct EncodedRow {
    board: Vec<f32>,
    globals: Vec<f32>,
    critic: Vec<f32>,
    units: Vec<f32>,
    positions: Vec<i64>,
    active: Vec<bool>,
}

fn encode_row(game: &Game, player: usize) -> EncodedRow {
    let mut row = EncodedRow {
        board: vec![0.0; BOARD_CHANNELS * BOARD_SIZE * BOARD_SIZE],
        globals: vec![0.0; GLOBAL_FEATURES],
        critic: vec![0.0; CRITIC_FEATURES],
        units: vec![0.0; MAX_UNITS * UNIT_FEATURES],
        positions: vec![0; MAX_UNITS * 2],
        active: vec![false; MAX_UNITS],
    };
    game.encode_player(
        player,
        &mut row.board,
        &mut row.globals,
        &mut row.critic,
        &mut row.units,
        &mut row.positions,
        &mut row.active,
    );
    row
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<BatchEnv>()?;
    module.add("MAX_UNITS", MAX_UNITS)?;
    module.add("MAX_MARKET_ORDERS", MAX_MARKET_ORDERS)?;
    module.add("N_UNIT_ACTIONS", UNIT_ACTIONS)?;
    module.add("N_MARKET_KINDS", MARKET_KINDS)?;
    module.add("N_QUANTITIES", MARKET_QUANTITIES)?;
    Ok(())
}
