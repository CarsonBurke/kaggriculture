use crate::core::{
    BOARD_CHANNELS, BOARD_SIZE, CRITIC_FEATURES, CROP_TOKEN_FIELDS, CROPS, CompactAction,
    FARM_TOKEN_FIELDS, GLOBAL_FEATURES, Game, GameConfig, MARKET_KINDS, MARKET_QUANTITIES,
    MAX_MARKET_ORDERS, MAX_SAFE_SEED, MAX_UNITS, PLAYERS, PRODUCT_TOKEN_FIELDS, PRODUCTS,
    SampledFactors, StepResult, TILE_CATEGORICAL, TILE_CONTINUOUS, TILE_TOKENS, TOWN_TOKEN_FIELDS,
    UNIT_ACTIONS, UNIT_CATEGORICAL, UNIT_CONTINUOUS, UNIT_FEATURES, UNIT_GATHERS,
};
use half::f16;
use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArray2, PyReadonlyArray3, PyReadwriteArray1, PyReadwriteArray2, PyReadwriteArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIndexError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

#[pyclass(name = "BatchEnv", unsendable)]
pub(crate) struct BatchEnv {
    games: Vec<Game>,
    sampled_scratch: Vec<SampledFactors>,
    results_scratch: Vec<StepResult>,
    /// Shaping potential of each game's current state, written by every step
    /// path so the next step's previous potential is a carry-forward instead
    /// of a recomputed liquidation walk.
    potential_cache: Vec<f32>,
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
        validate_seeds(seeds)?;
        let games: Vec<Game> = seeds
            .iter()
            .map(|&seed| Game::new(seed, GameConfig::default()))
            .collect();
        Ok(Self {
            sampled_scratch: (0..games.len() * PLAYERS)
                .map(|_| SampledFactors::default())
                .collect(),
            results_scratch: vec![StepResult::default(); games.len()],
            potential_cache: games.iter().map(Game::pair_potential).collect(),
            games,
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
        validate_seeds(seeds)?;
        for (game, &seed) in self.games.iter_mut().zip(seeds) {
            *game = Game::new(seed, GameConfig::default());
        }
        for (cached, game) in self.potential_cache.iter_mut().zip(&self.games) {
            *cached = game.pair_potential();
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
        let output = allocate_encoded_buffers(py, self.games.len())?;
        fill_encoded_output(py, &self.games, &output)?;
        Ok(output)
    }

    /// Allocate the exact output arrays expected by `encoded_into` once.
    fn encoded_buffers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        allocate_encoded_buffers(py, self.games.len())
    }

    /// Fill caller-owned, writable C-contiguous arrays without allocating.
    fn encoded_into(&self, py: Python<'_>, output: &Bound<'_, PyDict>) -> PyResult<()> {
        fill_encoded_output(py, &self.games, output)
    }

    /// Encode both seats' structured token bundles without Python objects.
    fn structured<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let output = allocate_structured_buffers(py, self.games.len())?;
        fill_structured_output(py, &self.games, &output)?;
        Ok(output)
    }

    /// Allocate the exact output arrays expected by `structured_into` once.
    fn structured_buffers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        allocate_structured_buffers(py, self.games.len())
    }

    /// Fill caller-owned, writable C-contiguous structured arrays without allocating.
    fn structured_into(&self, py: Python<'_>, output: &Bound<'_, PyDict>) -> PyResult<()> {
        fill_structured_output(py, &self.games, output)
    }

    /// Exact sequential masks for every supplied factor row, before stepping.
    fn factor_masks<'py>(
        &self,
        py: Python<'py>,
        unit_actions: PyReadonlyArray3<'py, u8>,
        market_kinds: PyReadonlyArray3<'py, u8>,
        market_quantities: PyReadonlyArray3<'py, u8>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let compact = extract_compact_actions(
            self.games.len(),
            unit_actions,
            market_kinds,
            market_quantities,
        )?;
        let masks = py.detach(|| {
            self.games
                .par_iter()
                .zip(compact.par_iter())
                .flat_map_iter(|(game, actions)| {
                    (0..PLAYERS).map(|player| game.factor_masks(player, &actions[player]))
                })
                .collect::<Vec<_>>()
        });
        let rows = masks.len();
        let output = PyDict::new(py);
        output.set_item(
            "unit_masks",
            Array3::from_shape_vec(
                (rows, MAX_UNITS, UNIT_ACTIONS),
                masks
                    .iter()
                    .flat_map(|mask| mask.unit.iter().copied())
                    .collect(),
            )
            .expect("unit mask shape is internal")
            .into_pyarray(py),
        )?;
        output.set_item(
            "market_kind_masks",
            Array3::from_shape_vec(
                (rows, MAX_MARKET_ORDERS, MARKET_KINDS),
                masks
                    .iter()
                    .flat_map(|mask| mask.market_kind.iter().copied())
                    .collect(),
            )
            .expect("market kind mask shape is internal")
            .into_pyarray(py),
        )?;
        output.set_item(
            "market_quantity_masks",
            Array3::from_shape_vec(
                (rows, MAX_MARKET_ORDERS, MARKET_QUANTITIES),
                masks
                    .iter()
                    .flat_map(|mask| mask.market_quantity.iter().copied())
                    .collect(),
            )
            .expect("market quantity mask shape is internal")
            .into_pyarray(py),
        )?;
        output.set_item(
            "unit_active",
            Array2::from_shape_vec(
                (rows, MAX_UNITS),
                masks.iter().flat_map(|mask| mask.unit_active).collect(),
            )
            .expect("unit active shape is internal")
            .into_pyarray(py),
        )?;
        output.set_item(
            "market_active",
            Array2::from_shape_vec(
                (rows, MAX_MARKET_ORDERS),
                masks.iter().flat_map(|mask| mask.market_active).collect(),
            )
            .expect("market active shape is internal")
            .into_pyarray(py),
        )?;
        output.set_item(
            "market_quantity_active",
            Array2::from_shape_vec(
                (rows, MAX_MARKET_ORDERS),
                masks
                    .iter()
                    .flat_map(|mask| mask.market_quantity_active)
                    .collect(),
            )
            .expect("quantity active shape is internal")
            .into_pyarray(py),
        )?;
        Ok(output)
    }

    /// Allocate the exact output arrays expected by `sample_and_step_into` once.
    fn sample_buffers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        allocate_sample_buffers(py, self.games.len())
    }

    /// Allocating convenience wrapper around `sample_and_step_into`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        unit_logits, market_kind_logits, market_quantity_context,
        quantity_kind_gate, quantity_values, quantity_bias, head_ids,
        unit_draws, market_kind_draws, market_quantity_draws,
        deterministic_rows, temperatures
    ))]
    fn sample_and_step<'py>(
        &mut self,
        py: Python<'py>,
        unit_logits: PyReadonlyArray3<'py, f32>,
        market_kind_logits: PyReadonlyArray3<'py, f32>,
        market_quantity_context: PyReadonlyArray3<'py, f32>,
        quantity_kind_gate: PyReadonlyArray3<'py, f32>,
        quantity_values: PyReadonlyArray3<'py, f32>,
        quantity_bias: PyReadonlyArray3<'py, f32>,
        head_ids: PyReadonlyArray1<'py, u16>,
        unit_draws: PyReadonlyArray2<'py, f32>,
        market_kind_draws: PyReadonlyArray2<'py, f32>,
        market_quantity_draws: PyReadonlyArray2<'py, f32>,
        deterministic_rows: PyReadonlyArray1<'py, bool>,
        temperatures: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let output = allocate_sample_buffers(py, self.games.len())?;
        self.sample_and_step_into(
            py,
            unit_logits,
            market_kind_logits,
            market_quantity_context,
            quantity_kind_gate,
            quantity_values,
            quantity_bias,
            head_ids,
            unit_draws,
            market_kind_draws,
            market_quantity_draws,
            deterministic_rows,
            temperatures,
            &output,
        )?;
        Ok(output)
    }

    /// Fused sequential masking, sampling, exact joint step, and direct output fill.
    ///
    /// Factor rows use game-major/player-minor order. Explicit draws keep the
    /// checkpointed NumPy RNG as the sole stochastic authority. Output arrays
    /// must come from `sample_buffers` or match its exact writable C layout.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        unit_logits, market_kind_logits, market_quantity_context,
        quantity_kind_gate, quantity_values, quantity_bias, head_ids,
        unit_draws, market_kind_draws, market_quantity_draws,
        deterministic_rows, temperatures, output
    ))]
    fn sample_and_step_into<'py>(
        &mut self,
        py: Python<'py>,
        unit_logits: PyReadonlyArray3<'py, f32>,
        market_kind_logits: PyReadonlyArray3<'py, f32>,
        market_quantity_context: PyReadonlyArray3<'py, f32>,
        quantity_kind_gate: PyReadonlyArray3<'py, f32>,
        quantity_values: PyReadonlyArray3<'py, f32>,
        quantity_bias: PyReadonlyArray3<'py, f32>,
        head_ids: PyReadonlyArray1<'py, u16>,
        unit_draws: PyReadonlyArray2<'py, f32>,
        market_kind_draws: PyReadonlyArray2<'py, f32>,
        market_quantity_draws: PyReadonlyArray2<'py, f32>,
        deterministic_rows: PyReadonlyArray1<'py, bool>,
        temperatures: PyReadonlyArray1<'py, f32>,
        output: &Bound<'py, PyDict>,
    ) -> PyResult<()> {
        let rows = self.games.len() * PLAYERS;
        macro_rules! require_c_input {
            ($array:ident, $name:literal) => {
                if !$array.is_c_contiguous() {
                    return Err(PyValueError::new_err(concat!(
                        $name,
                        " must be C-contiguous"
                    )));
                }
            };
        }
        ensure_shape(
            unit_logits.shape(),
            &[rows, MAX_UNITS, UNIT_ACTIONS],
            "unit_logits",
        )?;
        ensure_shape(
            market_kind_logits.shape(),
            &[rows, MAX_MARKET_ORDERS, MARKET_KINDS],
            "market_kind_logits",
        )?;
        let context_shape = market_quantity_context.shape();
        if context_shape.len() != 3
            || context_shape[0] != rows
            || context_shape[1] != MAX_MARKET_ORDERS
        {
            return Err(PyValueError::new_err(format!(
                "market_quantity_context shape {context_shape:?}, expected [{rows}, {MAX_MARKET_ORDERS}, rank]"
            )));
        }
        let rank = context_shape[2];
        let head_shape = quantity_kind_gate.shape();
        if head_shape.len() != 3
            || head_shape[1] != MARKET_KINDS
            || head_shape[2] != rank
            || head_shape[0] == 0
        {
            return Err(PyValueError::new_err(format!(
                "quantity_kind_gate shape {head_shape:?}, expected [heads, {MARKET_KINDS}, {rank}]"
            )));
        }
        let heads = head_shape[0];
        ensure_shape(
            quantity_values.shape(),
            &[heads, MARKET_QUANTITIES, rank],
            "quantity_values",
        )?;
        ensure_shape(
            quantity_bias.shape(),
            &[heads, MARKET_KINDS, MARKET_QUANTITIES],
            "quantity_bias",
        )?;
        ensure_shape(head_ids.shape(), &[rows], "head_ids")?;
        ensure_shape(deterministic_rows.shape(), &[rows], "deterministic_rows")?;
        ensure_shape(temperatures.shape(), &[rows], "temperatures")?;
        ensure_shape(unit_draws.shape(), &[rows, MAX_UNITS], "unit_draws")?;
        ensure_shape(
            market_kind_draws.shape(),
            &[rows, MAX_MARKET_ORDERS],
            "market_kind_draws",
        )?;
        ensure_shape(
            market_quantity_draws.shape(),
            &[rows, MAX_MARKET_ORDERS],
            "market_quantity_draws",
        )?;

        require_c_input!(unit_logits, "unit_logits");
        require_c_input!(market_kind_logits, "market_kind_logits");
        require_c_input!(market_quantity_context, "market_quantity_context");
        require_c_input!(quantity_kind_gate, "quantity_kind_gate");
        require_c_input!(quantity_values, "quantity_values");
        require_c_input!(quantity_bias, "quantity_bias");
        require_c_input!(head_ids, "head_ids");
        require_c_input!(unit_draws, "unit_draws");
        require_c_input!(market_kind_draws, "market_kind_draws");
        require_c_input!(market_quantity_draws, "market_quantity_draws");
        require_c_input!(deterministic_rows, "deterministic_rows");
        require_c_input!(temperatures, "temperatures");

        let unit_logits = unit_logits.as_slice()?;
        let kind_logits = market_kind_logits.as_slice()?;
        let quantity_context = market_quantity_context.as_slice()?;
        let kind_gate = quantity_kind_gate.as_slice()?;
        let quantity_values = quantity_values.as_slice()?;
        let quantity_bias = quantity_bias.as_slice()?;
        let head_ids = head_ids.as_slice()?;
        let unit_draws = unit_draws.as_slice()?;
        let kind_draws = market_kind_draws.as_slice()?;
        let quantity_draws = market_quantity_draws.as_slice()?;
        let deterministic_rows = deterministic_rows.as_slice()?;
        let temperatures = temperatures.as_slice()?;
        if head_ids.iter().any(|&head| usize::from(head) >= heads) {
            return Err(PyValueError::new_err(
                "head_ids contains an out-of-range head",
            ));
        }
        if temperatures
            .iter()
            .any(|&temperature| !temperature.is_finite() || temperature <= 0.0)
        {
            return Err(PyValueError::new_err(
                "temperatures must contain finite positive values",
            ));
        }
        for (name, values) in [
            ("unit_draws", unit_draws.iter()),
            ("market_kind_draws", kind_draws.iter()),
            ("market_quantity_draws", quantity_draws.iter()),
        ] {
            if values
                .clone()
                .any(|&value| !value.is_finite() || !(0.0..1.0).contains(&value))
            {
                return Err(PyValueError::new_err(format!(
                    "{name} must contain finite values in [0, 1)"
                )));
            }
        }

        let mut output_arrays = SampleOutputArrays::new(output, rows, self.games.len())?;
        let mut output_slices = output_arrays.slices()?;
        {
            let games = &self.games;
            let sampled = &mut self.sampled_scratch;
            py.detach(|| {
                sampled
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(row, output)| {
                        let game = &games[row / PLAYERS];
                        let player = row % PLAYERS;
                        let head_id = usize::from(head_ids[row]);
                        let gate_offset = head_id * MARKET_KINDS * rank;
                        let values_offset = head_id * MARKET_QUANTITIES * rank;
                        let bias_offset = head_id * MARKET_KINDS * MARKET_QUANTITIES;
                        let head = crate::core::QuantityHead {
                            rank,
                            kind_gate: &kind_gate[gate_offset..gate_offset + MARKET_KINDS * rank],
                            values: &quantity_values
                                [values_offset..values_offset + MARKET_QUANTITIES * rank],
                            bias: &quantity_bias
                                [bias_offset..bias_offset + MARKET_KINDS * MARKET_QUANTITIES],
                        };
                        let unit_offset = row * MAX_UNITS * UNIT_ACTIONS;
                        let kind_offset = row * MAX_MARKET_ORDERS * MARKET_KINDS;
                        let context_offset = row * MAX_MARKET_ORDERS * rank;
                        let unit_draw_offset = row * MAX_UNITS;
                        let market_draw_offset = row * MAX_MARKET_ORDERS;
                        *output = game.sample_factors(
                            player,
                            &unit_logits[unit_offset..unit_offset + MAX_UNITS * UNIT_ACTIONS],
                            &kind_logits
                                [kind_offset..kind_offset + MAX_MARKET_ORDERS * MARKET_KINDS],
                            &quantity_context
                                [context_offset..context_offset + MAX_MARKET_ORDERS * rank],
                            &head,
                            &unit_draws[unit_draw_offset..unit_draw_offset + MAX_UNITS],
                            &kind_draws[market_draw_offset..market_draw_offset + MAX_MARKET_ORDERS],
                            &quantity_draws
                                [market_draw_offset..market_draw_offset + MAX_MARKET_ORDERS],
                            deterministic_rows[row],
                            temperatures[row],
                        );
                    });
            });
        }
        {
            let sampled = &self.sampled_scratch;
            py.detach(|| {
                self.games
                    .par_iter_mut()
                    .zip(self.results_scratch.par_iter_mut())
                    .enumerate()
                    .for_each(|(game_index, (game, result))| {
                        let row = game_index * PLAYERS;
                        *result = game.step(&[sampled[row].action, sampled[row + 1].action]);
                    });
            });
        }
        fill_sample_step_output(
            &self.games,
            &self.sampled_scratch,
            &self.results_scratch,
            &mut self.potential_cache,
            &mut output_slices,
        );
        Ok(())
    }

    fn step_factors<'py>(
        &mut self,
        py: Python<'py>,
        unit_actions: PyReadonlyArray3<'py, u8>,
        market_kinds: PyReadonlyArray3<'py, u8>,
        market_quantities: PyReadonlyArray3<'py, u8>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let compact = extract_compact_actions(
            self.games.len(),
            unit_actions,
            market_kinds,
            market_quantities,
        )?;
        let previous_potentials = self.potential_cache.clone();
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
        let post_potentials: Vec<f32> = self.games.iter().map(Game::post_step_potential).collect();
        self.potential_cache.copy_from_slice(&post_potentials);
        let shaped: Vec<f32> = previous_potentials
            .iter()
            .zip(post_potentials.iter())
            .flat_map(|(&previous, &post)| {
                let reward_zero = post - previous;
                [reward_zero, -reward_zero]
            })
            .collect();
        output.set_item(
            "previous_potentials",
            Array1::from_vec(previous_potentials).into_pyarray(py),
        )?;
        output.set_item(
            "potentials",
            Array1::from_vec(post_potentials).into_pyarray(py),
        )?;
        output.set_item(
            "shaped_rewards",
            Array2::from_shape_vec((self.games.len(), PLAYERS), shaped)
                .expect("shaped reward shape is internal")
                .into_pyarray(py),
        )?;
        Ok(output)
    }
}

fn allocate_encoded_buffers<'py>(py: Python<'py>, batch: usize) -> PyResult<Bound<'py, PyDict>> {
    let rows = batch * PLAYERS;
    let output = PyDict::new(py);
    output.set_item(
        "board",
        PyArray4::<f16>::zeros(py, [rows, BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE], false),
    )?;
    output.set_item(
        "global_features",
        PyArray2::<f16>::zeros(py, [rows, GLOBAL_FEATURES], false),
    )?;
    output.set_item(
        "critic_features",
        PyArray2::<f16>::zeros(py, [rows, CRITIC_FEATURES], false),
    )?;
    output.set_item(
        "units",
        PyArray3::<f16>::zeros(py, [rows, MAX_UNITS, UNIT_FEATURES], false),
    )?;
    output.set_item(
        "unit_positions",
        PyArray3::<i64>::zeros(py, [rows, MAX_UNITS, 2], false),
    )?;
    output.set_item(
        "unit_active",
        PyArray2::<bool>::zeros(py, [rows, MAX_UNITS], false),
    )?;
    output.set_item("potentials", PyArray1::<f32>::zeros(py, batch, false))?;
    Ok(output)
}

fn allocate_structured_buffers<'py>(py: Python<'py>, batch: usize) -> PyResult<Bound<'py, PyDict>> {
    let rows = batch * PLAYERS;
    let output = PyDict::new(py);
    output.set_item(
        "tile_categorical",
        PyArray3::<i8>::zeros(py, [rows, TILE_TOKENS, TILE_CATEGORICAL], false),
    )?;
    output.set_item(
        "tile_continuous",
        PyArray3::<f16>::zeros(py, [rows, TILE_TOKENS, TILE_CONTINUOUS], false),
    )?;
    output.set_item(
        "unit_categorical",
        PyArray3::<i8>::zeros(py, [rows, MAX_UNITS, UNIT_CATEGORICAL], false),
    )?;
    output.set_item(
        "unit_continuous",
        PyArray3::<f16>::zeros(py, [rows, MAX_UNITS, UNIT_CONTINUOUS], false),
    )?;
    output.set_item(
        "unit_active",
        PyArray2::<bool>::zeros(py, [rows, MAX_UNITS], false),
    )?;
    output.set_item(
        "unit_tile_gather",
        PyArray3::<i8>::zeros(py, [rows, MAX_UNITS, UNIT_GATHERS], false),
    )?;
    output.set_item(
        "unit_tile_gather_valid",
        PyArray3::<bool>::zeros(py, [rows, MAX_UNITS, UNIT_GATHERS], false),
    )?;
    output.set_item(
        "products",
        PyArray3::<f16>::zeros(py, [rows, PRODUCTS, PRODUCT_TOKEN_FIELDS], false),
    )?;
    output.set_item(
        "crops",
        PyArray3::<f16>::zeros(py, [rows, CROPS, CROP_TOKEN_FIELDS], false),
    )?;
    output.set_item(
        "farms",
        PyArray3::<f16>::zeros(py, [rows, PLAYERS, FARM_TOKEN_FIELDS], false),
    )?;
    output.set_item(
        "town",
        PyArray2::<f16>::zeros(py, [rows, TOWN_TOKEN_FIELDS], false),
    )?;
    output.set_item("potentials", PyArray1::<f32>::zeros(py, batch, false))?;
    Ok(output)
}

fn fill_structured_output(
    py: Python<'_>,
    games: &[Game],
    output: &Bound<'_, PyDict>,
) -> PyResult<()> {
    let rows = games.len() * PLAYERS;
    macro_rules! output_array {
        ($name:literal, $type:ty, $shape:expr) => {{
            let array = required_output(output, $name)?.cast_into::<$type>()?;
            ensure_shape(array.shape(), &$shape, concat!("output ", $name))?;
            if !array.is_c_contiguous() {
                return Err(non_contiguous($name));
            }
            array.try_readwrite()?
        }};
    }
    let mut tile_categorical = output_array!(
        "tile_categorical",
        PyArray3<i8>,
        [rows, TILE_TOKENS, TILE_CATEGORICAL]
    );
    let mut tile_continuous = output_array!(
        "tile_continuous",
        PyArray3<f16>,
        [rows, TILE_TOKENS, TILE_CONTINUOUS]
    );
    let mut unit_categorical = output_array!(
        "unit_categorical",
        PyArray3<i8>,
        [rows, MAX_UNITS, UNIT_CATEGORICAL]
    );
    let mut unit_continuous = output_array!(
        "unit_continuous",
        PyArray3<f16>,
        [rows, MAX_UNITS, UNIT_CONTINUOUS]
    );
    let mut unit_active = output_array!("unit_active", PyArray2<bool>, [rows, MAX_UNITS]);
    let mut unit_tile_gather = output_array!(
        "unit_tile_gather",
        PyArray3<i8>,
        [rows, MAX_UNITS, UNIT_GATHERS]
    );
    let mut unit_tile_gather_valid = output_array!(
        "unit_tile_gather_valid",
        PyArray3<bool>,
        [rows, MAX_UNITS, UNIT_GATHERS]
    );
    let mut products = output_array!(
        "products",
        PyArray3<f16>,
        [rows, PRODUCTS, PRODUCT_TOKEN_FIELDS]
    );
    let mut crops = output_array!("crops", PyArray3<f16>, [rows, CROPS, CROP_TOKEN_FIELDS]);
    let mut farms = output_array!("farms", PyArray3<f16>, [rows, PLAYERS, FARM_TOKEN_FIELDS]);
    let mut town = output_array!("town", PyArray2<f16>, [rows, TOWN_TOKEN_FIELDS]);
    let mut potentials = output_array!("potentials", PyArray1<f32>, [games.len()]);

    let tile_categorical = tile_categorical
        .as_slice_mut()
        .map_err(|_| non_contiguous("tile_categorical"))?;
    let tile_continuous = tile_continuous
        .as_slice_mut()
        .map_err(|_| non_contiguous("tile_continuous"))?;
    let unit_categorical = unit_categorical
        .as_slice_mut()
        .map_err(|_| non_contiguous("unit_categorical"))?;
    let unit_continuous = unit_continuous
        .as_slice_mut()
        .map_err(|_| non_contiguous("unit_continuous"))?;
    let unit_active = unit_active
        .as_slice_mut()
        .map_err(|_| non_contiguous("unit_active"))?;
    let unit_tile_gather = unit_tile_gather
        .as_slice_mut()
        .map_err(|_| non_contiguous("unit_tile_gather"))?;
    let unit_tile_gather_valid = unit_tile_gather_valid
        .as_slice_mut()
        .map_err(|_| non_contiguous("unit_tile_gather_valid"))?;
    let products = products
        .as_slice_mut()
        .map_err(|_| non_contiguous("products"))?;
    let crops = crops.as_slice_mut().map_err(|_| non_contiguous("crops"))?;
    let farms = farms.as_slice_mut().map_err(|_| non_contiguous("farms"))?;
    let town = town.as_slice_mut().map_err(|_| non_contiguous("town"))?;
    let potentials = potentials
        .as_slice_mut()
        .map_err(|_| non_contiguous("potentials"))?;

    const TILE_CATEGORICAL_VALUES: usize = TILE_TOKENS * TILE_CATEGORICAL;
    const TILE_CONTINUOUS_VALUES: usize = TILE_TOKENS * TILE_CONTINUOUS;
    const UNIT_CATEGORICAL_VALUES: usize = MAX_UNITS * UNIT_CATEGORICAL;
    const UNIT_CONTINUOUS_VALUES: usize = MAX_UNITS * UNIT_CONTINUOUS;
    const UNIT_GATHER_VALUES: usize = MAX_UNITS * UNIT_GATHERS;
    const PRODUCT_VALUES: usize = PRODUCTS * PRODUCT_TOKEN_FIELDS;
    const CROP_VALUES: usize = CROPS * CROP_TOKEN_FIELDS;
    const FARM_VALUES: usize = PLAYERS * FARM_TOKEN_FIELDS;
    py.detach(|| {
        tile_categorical
            .par_chunks_mut(TILE_CATEGORICAL_VALUES)
            .zip(tile_continuous.par_chunks_mut(TILE_CONTINUOUS_VALUES))
            .zip(unit_categorical.par_chunks_mut(UNIT_CATEGORICAL_VALUES))
            .zip(unit_continuous.par_chunks_mut(UNIT_CONTINUOUS_VALUES))
            .zip(unit_active.par_chunks_mut(MAX_UNITS))
            .zip(unit_tile_gather.par_chunks_mut(UNIT_GATHER_VALUES))
            .zip(unit_tile_gather_valid.par_chunks_mut(UNIT_GATHER_VALUES))
            .zip(products.par_chunks_mut(PRODUCT_VALUES))
            .zip(crops.par_chunks_mut(CROP_VALUES))
            .zip(farms.par_chunks_mut(FARM_VALUES))
            .zip(town.par_chunks_mut(TOWN_TOKEN_FIELDS))
            .enumerate()
            .for_each(
                |(
                    row,
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (tile_categorical, tile_continuous),
                                                        unit_categorical,
                                                    ),
                                                    unit_continuous,
                                                ),
                                                unit_active,
                                            ),
                                            unit_tile_gather,
                                        ),
                                        unit_tile_gather_valid,
                                    ),
                                    products,
                                ),
                                crops,
                            ),
                            farms,
                        ),
                        town,
                    ),
                )| {
                    let mut tile_continuous_f32 = [0.0f32; TILE_CONTINUOUS_VALUES];
                    let mut unit_continuous_f32 = [0.0f32; UNIT_CONTINUOUS_VALUES];
                    let mut products_f32 = [0.0f32; PRODUCT_VALUES];
                    let mut crops_f32 = [0.0f32; CROP_VALUES];
                    let mut farms_f32 = [0.0f32; FARM_VALUES];
                    let mut town_f32 = [0.0f32; TOWN_TOKEN_FIELDS];
                    games[row / PLAYERS].encode_player_structured(
                        row % PLAYERS,
                        tile_categorical,
                        &mut tile_continuous_f32,
                        unit_categorical,
                        &mut unit_continuous_f32,
                        unit_active,
                        unit_tile_gather,
                        unit_tile_gather_valid,
                        &mut products_f32,
                        &mut crops_f32,
                        &mut farms_f32,
                        &mut town_f32,
                    );
                    for (target, value) in tile_continuous.iter_mut().zip(tile_continuous_f32) {
                        *target = f16::from_f32(value);
                    }
                    for (target, value) in unit_continuous.iter_mut().zip(unit_continuous_f32) {
                        *target = f16::from_f32(value);
                    }
                    for (target, value) in products.iter_mut().zip(products_f32) {
                        *target = f16::from_f32(value);
                    }
                    for (target, value) in crops.iter_mut().zip(crops_f32) {
                        *target = f16::from_f32(value);
                    }
                    for (target, value) in farms.iter_mut().zip(farms_f32) {
                        *target = f16::from_f32(value);
                    }
                    for (target, value) in town.iter_mut().zip(town_f32) {
                        *target = f16::from_f32(value);
                    }
                },
            );
        potentials
            .par_iter_mut()
            .zip(games.par_iter())
            .for_each(|(output, game)| *output = game.pair_potential());
    });
    Ok(())
}

fn allocate_sample_buffers<'py>(py: Python<'py>, batch: usize) -> PyResult<Bound<'py, PyDict>> {
    let rows = batch * PLAYERS;
    let output = PyDict::new(py);
    output.set_item(
        "unit_actions",
        PyArray2::<u8>::zeros(py, [rows, MAX_UNITS], false),
    )?;
    for name in ["market_kinds", "market_quantities"] {
        output.set_item(
            name,
            PyArray2::<u8>::zeros(py, [rows, MAX_MARKET_ORDERS], false),
        )?;
    }
    output.set_item(
        "unit_masks",
        PyArray3::<bool>::zeros(py, [rows, MAX_UNITS, UNIT_ACTIONS], false),
    )?;
    output.set_item(
        "market_kind_masks",
        PyArray3::<bool>::zeros(py, [rows, MAX_MARKET_ORDERS, MARKET_KINDS], false),
    )?;
    output.set_item(
        "market_quantity_masks",
        PyArray3::<bool>::zeros(py, [rows, MAX_MARKET_ORDERS, MARKET_QUANTITIES], false),
    )?;
    output.set_item(
        "unit_active",
        PyArray2::<bool>::zeros(py, [rows, MAX_UNITS], false),
    )?;
    for name in ["market_active", "market_quantity_active"] {
        output.set_item(
            name,
            PyArray2::<bool>::zeros(py, [rows, MAX_MARKET_ORDERS], false),
        )?;
    }
    output.set_item(
        "unit_logprobs",
        PyArray2::<f32>::zeros(py, [rows, MAX_UNITS], false),
    )?;
    for name in ["market_kind_logprobs", "market_quantity_logprobs"] {
        output.set_item(
            name,
            PyArray2::<f32>::zeros(py, [rows, MAX_MARKET_ORDERS], false),
        )?;
    }
    output.set_item("entropy", PyArray1::<f32>::zeros(py, rows, false))?;
    for name in ["rewards", "final_money", "shaped_rewards"] {
        output.set_item(name, PyArray2::<f32>::zeros(py, [batch, PLAYERS], false))?;
    }
    output.set_item("dones", PyArray1::<bool>::zeros(py, batch, false))?;
    for name in ["previous_potentials", "potentials"] {
        output.set_item(name, PyArray1::<f32>::zeros(py, batch, false))?;
    }
    Ok(output)
}

struct SampleOutputArrays<'py> {
    unit_actions: PyReadwriteArray2<'py, u8>,
    market_kinds: PyReadwriteArray2<'py, u8>,
    market_quantities: PyReadwriteArray2<'py, u8>,
    unit_masks: PyReadwriteArray3<'py, bool>,
    market_kind_masks: PyReadwriteArray3<'py, bool>,
    market_quantity_masks: PyReadwriteArray3<'py, bool>,
    unit_active: PyReadwriteArray2<'py, bool>,
    market_active: PyReadwriteArray2<'py, bool>,
    market_quantity_active: PyReadwriteArray2<'py, bool>,
    unit_logprobs: PyReadwriteArray2<'py, f32>,
    market_kind_logprobs: PyReadwriteArray2<'py, f32>,
    market_quantity_logprobs: PyReadwriteArray2<'py, f32>,
    entropy: PyReadwriteArray1<'py, f32>,
    rewards: PyReadwriteArray2<'py, f32>,
    money: PyReadwriteArray2<'py, f32>,
    dones: PyReadwriteArray1<'py, bool>,
    previous: PyReadwriteArray1<'py, f32>,
    potentials: PyReadwriteArray1<'py, f32>,
    shaped: PyReadwriteArray2<'py, f32>,
}

impl<'py> SampleOutputArrays<'py> {
    fn new(output: &Bound<'py, PyDict>, rows: usize, batch: usize) -> PyResult<Self> {
        macro_rules! output_array {
            ($name:literal, $type:ty, $shape:expr) => {{
                let array = required_output(output, $name)?.cast_into::<$type>()?;
                ensure_shape(array.shape(), &$shape, concat!("output ", $name))?;
                if !array.is_c_contiguous() {
                    return Err(non_contiguous($name));
                }
                array.try_readwrite()?
            }};
        }
        Ok(Self {
            unit_actions: output_array!("unit_actions", PyArray2<u8>, [rows, MAX_UNITS]),
            market_kinds: output_array!("market_kinds", PyArray2<u8>, [rows, MAX_MARKET_ORDERS]),
            market_quantities: output_array!(
                "market_quantities",
                PyArray2<u8>,
                [rows, MAX_MARKET_ORDERS]
            ),
            unit_masks: output_array!(
                "unit_masks",
                PyArray3<bool>,
                [rows, MAX_UNITS, UNIT_ACTIONS]
            ),
            market_kind_masks: output_array!(
                "market_kind_masks",
                PyArray3<bool>,
                [rows, MAX_MARKET_ORDERS, MARKET_KINDS]
            ),
            market_quantity_masks: output_array!(
                "market_quantity_masks",
                PyArray3<bool>,
                [rows, MAX_MARKET_ORDERS, MARKET_QUANTITIES]
            ),
            unit_active: output_array!("unit_active", PyArray2<bool>, [rows, MAX_UNITS]),
            market_active: output_array!(
                "market_active",
                PyArray2<bool>,
                [rows, MAX_MARKET_ORDERS]
            ),
            market_quantity_active: output_array!(
                "market_quantity_active",
                PyArray2<bool>,
                [rows, MAX_MARKET_ORDERS]
            ),
            unit_logprobs: output_array!("unit_logprobs", PyArray2<f32>, [rows, MAX_UNITS]),
            market_kind_logprobs: output_array!(
                "market_kind_logprobs",
                PyArray2<f32>,
                [rows, MAX_MARKET_ORDERS]
            ),
            market_quantity_logprobs: output_array!(
                "market_quantity_logprobs",
                PyArray2<f32>,
                [rows, MAX_MARKET_ORDERS]
            ),
            entropy: output_array!("entropy", PyArray1<f32>, [rows]),
            rewards: output_array!("rewards", PyArray2<f32>, [batch, PLAYERS]),
            money: output_array!("final_money", PyArray2<f32>, [batch, PLAYERS]),
            dones: output_array!("dones", PyArray1<bool>, [batch]),
            previous: output_array!("previous_potentials", PyArray1<f32>, [batch]),
            potentials: output_array!("potentials", PyArray1<f32>, [batch]),
            shaped: output_array!("shaped_rewards", PyArray2<f32>, [batch, PLAYERS]),
        })
    }

    fn slices(&mut self) -> PyResult<SampleOutputSlices<'_>> {
        let Self {
            unit_actions,
            market_kinds,
            market_quantities,
            unit_masks,
            market_kind_masks,
            market_quantity_masks,
            unit_active,
            market_active,
            market_quantity_active,
            unit_logprobs,
            market_kind_logprobs,
            market_quantity_logprobs,
            entropy,
            rewards,
            money,
            dones,
            previous,
            potentials,
            shaped,
        } = self;
        Ok(SampleOutputSlices {
            unit_actions: unit_actions
                .as_slice_mut()
                .map_err(|_| non_contiguous("unit_actions"))?,
            market_kinds: market_kinds
                .as_slice_mut()
                .map_err(|_| non_contiguous("market_kinds"))?,
            market_quantities: market_quantities
                .as_slice_mut()
                .map_err(|_| non_contiguous("market_quantities"))?,
            unit_masks: unit_masks
                .as_slice_mut()
                .map_err(|_| non_contiguous("unit_masks"))?,
            market_kind_masks: market_kind_masks
                .as_slice_mut()
                .map_err(|_| non_contiguous("market_kind_masks"))?,
            market_quantity_masks: market_quantity_masks
                .as_slice_mut()
                .map_err(|_| non_contiguous("market_quantity_masks"))?,
            unit_active: unit_active
                .as_slice_mut()
                .map_err(|_| non_contiguous("unit_active"))?,
            market_active: market_active
                .as_slice_mut()
                .map_err(|_| non_contiguous("market_active"))?,
            market_quantity_active: market_quantity_active
                .as_slice_mut()
                .map_err(|_| non_contiguous("market_quantity_active"))?,
            unit_logprobs: unit_logprobs
                .as_slice_mut()
                .map_err(|_| non_contiguous("unit_logprobs"))?,
            market_kind_logprobs: market_kind_logprobs
                .as_slice_mut()
                .map_err(|_| non_contiguous("market_kind_logprobs"))?,
            market_quantity_logprobs: market_quantity_logprobs
                .as_slice_mut()
                .map_err(|_| non_contiguous("market_quantity_logprobs"))?,
            entropy: entropy
                .as_slice_mut()
                .map_err(|_| non_contiguous("entropy"))?,
            rewards: rewards
                .as_slice_mut()
                .map_err(|_| non_contiguous("rewards"))?,
            money: money
                .as_slice_mut()
                .map_err(|_| non_contiguous("final_money"))?,
            dones: dones.as_slice_mut().map_err(|_| non_contiguous("dones"))?,
            previous: previous
                .as_slice_mut()
                .map_err(|_| non_contiguous("previous_potentials"))?,
            potentials: potentials
                .as_slice_mut()
                .map_err(|_| non_contiguous("potentials"))?,
            shaped: shaped
                .as_slice_mut()
                .map_err(|_| non_contiguous("shaped_rewards"))?,
        })
    }
}

struct SampleOutputSlices<'a> {
    unit_actions: &'a mut [u8],
    market_kinds: &'a mut [u8],
    market_quantities: &'a mut [u8],
    unit_masks: &'a mut [bool],
    market_kind_masks: &'a mut [bool],
    market_quantity_masks: &'a mut [bool],
    unit_active: &'a mut [bool],
    market_active: &'a mut [bool],
    market_quantity_active: &'a mut [bool],
    unit_logprobs: &'a mut [f32],
    market_kind_logprobs: &'a mut [f32],
    market_quantity_logprobs: &'a mut [f32],
    entropy: &'a mut [f32],
    rewards: &'a mut [f32],
    money: &'a mut [f32],
    dones: &'a mut [bool],
    previous: &'a mut [f32],
    potentials: &'a mut [f32],
    shaped: &'a mut [f32],
}

fn required_output<'py>(output: &Bound<'py, PyDict>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    output
        .get_item(name)?
        .ok_or_else(|| PyKeyError::new_err(format!("missing output array {name:?}")))
}

fn non_contiguous(name: &str) -> PyErr {
    PyValueError::new_err(format!(
        "output array {name:?} must be writable and C-contiguous"
    ))
}

fn fill_encoded_output(py: Python<'_>, games: &[Game], output: &Bound<'_, PyDict>) -> PyResult<()> {
    let rows = games.len() * PLAYERS;
    macro_rules! output_array {
        ($name:literal, $type:ty, $shape:expr) => {{
            let array = required_output(output, $name)?.cast_into::<$type>()?;
            ensure_shape(array.shape(), &$shape, concat!("output ", $name))?;
            if !array.is_c_contiguous() {
                return Err(non_contiguous($name));
            }
            array.try_readwrite()?
        }};
    }
    let mut board = output_array!(
        "board",
        PyArray4<f16>,
        [rows, BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE]
    );
    let mut globals = output_array!("global_features", PyArray2<f16>, [rows, GLOBAL_FEATURES]);
    let mut critic = output_array!("critic_features", PyArray2<f16>, [rows, CRITIC_FEATURES]);
    let mut units = output_array!("units", PyArray3<f16>, [rows, MAX_UNITS, UNIT_FEATURES]);
    let mut positions = output_array!("unit_positions", PyArray3<i64>, [rows, MAX_UNITS, 2]);
    let mut active = output_array!("unit_active", PyArray2<bool>, [rows, MAX_UNITS]);
    let mut potentials = output_array!("potentials", PyArray1<f32>, [games.len()]);

    let board = board.as_slice_mut().map_err(|_| non_contiguous("board"))?;
    let globals = globals
        .as_slice_mut()
        .map_err(|_| non_contiguous("global_features"))?;
    let critic = critic
        .as_slice_mut()
        .map_err(|_| non_contiguous("critic_features"))?;
    let units = units.as_slice_mut().map_err(|_| non_contiguous("units"))?;
    let positions = positions
        .as_slice_mut()
        .map_err(|_| non_contiguous("unit_positions"))?;
    let active = active
        .as_slice_mut()
        .map_err(|_| non_contiguous("unit_active"))?;
    let potentials = potentials
        .as_slice_mut()
        .map_err(|_| non_contiguous("potentials"))?;

    const BOARD_VALUES: usize = BOARD_CHANNELS * BOARD_SIZE * BOARD_SIZE;
    const UNIT_VALUES: usize = MAX_UNITS * UNIT_FEATURES;
    py.detach(|| {
        board
            .par_chunks_mut(BOARD_VALUES)
            .zip(globals.par_chunks_mut(GLOBAL_FEATURES))
            .zip(critic.par_chunks_mut(CRITIC_FEATURES))
            .zip(units.par_chunks_mut(UNIT_VALUES))
            .zip(positions.par_chunks_mut(MAX_UNITS * 2))
            .zip(active.par_chunks_mut(MAX_UNITS))
            .enumerate()
            .for_each(
                |(row, (((((board, globals), critic), units), positions), active))| {
                    let mut board_f32 = [0.0; BOARD_VALUES];
                    let mut globals_f32 = [0.0; GLOBAL_FEATURES];
                    let mut critic_f32 = [0.0; CRITIC_FEATURES];
                    let mut units_f32 = [0.0; UNIT_VALUES];
                    games[row / PLAYERS].encode_player(
                        row % PLAYERS,
                        &mut board_f32,
                        &mut globals_f32,
                        &mut critic_f32,
                        &mut units_f32,
                        positions,
                        active,
                    );
                    for (target, value) in board.iter_mut().zip(board_f32) {
                        *target = f16::from_f32(value);
                    }
                    for (target, value) in globals.iter_mut().zip(globals_f32) {
                        *target = f16::from_f32(value);
                    }
                    for (target, value) in critic.iter_mut().zip(critic_f32) {
                        *target = f16::from_f32(value);
                    }
                    for (target, value) in units.iter_mut().zip(units_f32) {
                        *target = f16::from_f32(value);
                    }
                },
            );
        potentials
            .par_iter_mut()
            .zip(games.par_iter())
            .for_each(|(output, game)| *output = game.pair_potential());
    });
    Ok(())
}

fn ensure_shape(actual: &[usize], expected: &[usize], name: &str) -> PyResult<()> {
    if actual != expected {
        return Err(PyValueError::new_err(format!(
            "{name} shape {actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn validate_seeds(seeds: &[u64]) -> PyResult<()> {
    if seeds.iter().any(|&seed| seed > MAX_SAFE_SEED) {
        return Err(PyValueError::new_err(format!(
            "seed exceeds exact CPython-compatible maximum {MAX_SAFE_SEED}"
        )));
    }
    Ok(())
}

fn extract_compact_actions(
    games: usize,
    unit_actions: PyReadonlyArray3<'_, u8>,
    market_kinds: PyReadonlyArray3<'_, u8>,
    market_quantities: PyReadonlyArray3<'_, u8>,
) -> PyResult<Vec<[CompactAction; PLAYERS]>> {
    let expected_units = [games, PLAYERS, MAX_UNITS];
    let expected_market = [games, PLAYERS, MAX_MARKET_ORDERS];
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
    Ok((0..games)
        .map(|game| {
            std::array::from_fn(|player| CompactAction {
                units: std::array::from_fn(|unit| units[[game, player, unit]]),
                market_kinds: std::array::from_fn(|slot| kinds[[game, player, slot]]),
                market_quantities: std::array::from_fn(|slot| quantities[[game, player, slot]]),
            })
        })
        .collect())
}

fn fill_sample_step_output(
    games: &[Game],
    sampled: &[SampledFactors],
    results: &[StepResult],
    potential_cache: &mut [f32],
    output: &mut SampleOutputSlices<'_>,
) {
    let SampleOutputSlices {
        unit_actions,
        market_kinds,
        market_quantities,
        unit_masks,
        market_kind_masks,
        market_quantity_masks,
        unit_active,
        market_active,
        market_quantity_active,
        unit_logprobs,
        market_kind_logprobs,
        market_quantity_logprobs,
        entropy,
        rewards,
        money,
        dones,
        previous,
        potentials,
        shaped,
    } = output;

    for (row_index, row) in sampled.iter().enumerate() {
        let unit_offset = row_index * MAX_UNITS;
        let market_offset = row_index * MAX_MARKET_ORDERS;
        let unit_mask_offset = row_index * MAX_UNITS * UNIT_ACTIONS;
        let kind_mask_offset = row_index * MAX_MARKET_ORDERS * MARKET_KINDS;
        let quantity_mask_offset = row_index * MAX_MARKET_ORDERS * MARKET_QUANTITIES;
        unit_actions[unit_offset..unit_offset + MAX_UNITS].copy_from_slice(&row.action.units);
        market_kinds[market_offset..market_offset + MAX_MARKET_ORDERS]
            .copy_from_slice(&row.action.market_kinds);
        market_quantities[market_offset..market_offset + MAX_MARKET_ORDERS]
            .copy_from_slice(&row.action.market_quantities);
        unit_masks[unit_mask_offset..unit_mask_offset + row.masks.unit.len()]
            .copy_from_slice(&row.masks.unit);
        market_kind_masks[kind_mask_offset..kind_mask_offset + row.masks.market_kind.len()]
            .copy_from_slice(&row.masks.market_kind);
        market_quantity_masks
            [quantity_mask_offset..quantity_mask_offset + row.masks.market_quantity.len()]
            .copy_from_slice(&row.masks.market_quantity);
        unit_active[unit_offset..unit_offset + MAX_UNITS].copy_from_slice(&row.masks.unit_active);
        market_active[market_offset..market_offset + MAX_MARKET_ORDERS]
            .copy_from_slice(&row.masks.market_active);
        market_quantity_active[market_offset..market_offset + MAX_MARKET_ORDERS]
            .copy_from_slice(&row.masks.market_quantity_active);
        unit_logprobs[unit_offset..unit_offset + MAX_UNITS].copy_from_slice(&row.unit_logprobs);
        market_kind_logprobs[market_offset..market_offset + MAX_MARKET_ORDERS]
            .copy_from_slice(&row.market_kind_logprobs);
        market_quantity_logprobs[market_offset..market_offset + MAX_MARKET_ORDERS]
            .copy_from_slice(&row.market_quantity_logprobs);
        entropy[row_index] = row.mean_entropy;
    }
    for (game_index, ((game, result), cached)) in
        games.iter().zip(results).zip(potential_cache).enumerate()
    {
        let offset = game_index * PLAYERS;
        rewards[offset..offset + PLAYERS].copy_from_slice(&result.rewards);
        money[offset..offset + PLAYERS].copy_from_slice(&result.money);
        dones[game_index] = result.done;
        let pre = *cached;
        previous[game_index] = pre;
        let post = game.post_step_potential();
        potentials[game_index] = post;
        *cached = post;
        let reward_zero = post - pre;
        shaped[offset] = reward_zero;
        shaped[offset + 1] = -reward_zero;
    }
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
