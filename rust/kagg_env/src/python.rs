use crate::core::{
    ANIMAL_TOKEN_FIELDS, ANIMALS, BOARD_CHANNELS, BOARD_SIZE, BuiltinAgent, CRITIC_FEATURES,
    CROP_TOKEN_FIELDS, CROPS, CompactAction, FARM_TOKEN_FIELDS, GLOBAL_FEATURES, Game, GameConfig,
    MARKET_KINDS, MARKET_QUANTITIES, MAX_MARKET_ORDERS, MAX_SAFE_SEED, MAX_UNITS,
    OBSERVATION_SCHEMA_VERSION, PLAYERS, POLICY_LEDGER_SCHEMA_VERSION, POLICY_LEDGER_WIDTH,
    PRODUCT_TOKEN_FIELDS, PRODUCTS, PyRandom, SampledFactors, StepResult, TILE_CATEGORICAL,
    TILE_CONTINUOUS, TILE_TOKENS, TOWN_TOKEN_FIELDS, UNIT_ACTIONS, UNIT_CATEGORICAL,
    UNIT_CONTINUOUS, UNIT_FEATURES, UNIT_GATHERS, V27State,
};
use crate::v27_script::{V27_SOURCE_NAME, V27_SOURCE_SHA256, V27_STEPS};
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

#[pyclass(name = "BatchEnv")]
pub(crate) struct BatchEnv {
    games: Vec<Game>,
    sampled_scratch: Vec<SampledFactors>,
    results_scratch: Vec<StepResult>,
    /// Shaping potential of each game's current state. Every step path updates
    /// it, so the hot sampler does not repeat the liquidation walk.
    potential_cache: Vec<f32>,
    /// Per-seat memory for the scripted v27 opponent, one row per game seat in
    /// the same order as `sampled_scratch`. It needs no reset hook: the agent
    /// clears its own row when the step index restarts, exactly as the
    /// reference's module-level state does.
    v27_states: Vec<V27State>,
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
            v27_states: vec![V27State::default(); games.len() * PLAYERS],
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

    /// Allocate a caller-owned exact policy-state buffer in game/seat order.
    fn policy_ledger_buffers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i64>> {
        PyArray2::zeros(py, [self.games.len() * PLAYERS, POLICY_LEDGER_WIDTH], false)
    }

    fn policy_ledger<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let output = self.policy_ledger_buffers(py);
        self.policy_ledger_into(py, &output)?;
        Ok(output)
    }

    /// Fill reusable pinned-host-compatible memory without a Python observation.
    fn policy_ledger_into(
        &self,
        py: Python<'_>,
        output: &Bound<'_, PyArray2<i64>>,
    ) -> PyResult<()> {
        ensure_shape(
            output.shape(),
            &[self.games.len() * PLAYERS, POLICY_LEDGER_WIDTH],
            "policy ledger",
        )?;
        if !output.is_c_contiguous() {
            return Err(non_contiguous("policy ledger"));
        }
        let mut writable = output.try_readwrite()?;
        let values = writable
            .as_slice_mut()
            .map_err(|_| non_contiguous("policy ledger"))?;
        py.detach(|| {
            self.games
                .par_iter()
                .zip(values.par_chunks_mut(PLAYERS * POLICY_LEDGER_WIDTH))
                .for_each(|(game, pair)| {
                    for player in 0..PLAYERS {
                        game.encode_policy_ledger(
                            player,
                            &mut pair
                                [player * POLICY_LEDGER_WIDTH..(player + 1) * POLICY_LEDGER_WIDTH],
                        );
                    }
                });
        });
        Ok(())
    }

    /// Exact integer quotes generated by the engine, including ties-even and floor.
    /// The bound limits one table to the full default-game reachable inventory
    /// range; callers can request smaller tables for independent correctness tests.
    #[staticmethod]
    fn policy_market_prices<'py>(
        py: Python<'py>,
        minimum: i32,
        maximum: i32,
    ) -> PyResult<Bound<'py, PyArray2<i32>>> {
        let width = i64::from(maximum) - i64::from(minimum) + 1;
        if minimum < -1_444_000 || maximum > 1_452_000 || width <= 0 {
            return Err(PyValueError::new_err(
                "unsupported policy price-table bounds",
            ));
        }
        let width = width as usize;
        let mut values = vec![0i32; PRODUCTS * width];
        py.detach(|| {
            values
                .par_chunks_mut(width)
                .enumerate()
                .try_for_each(|(item, row)| {
                    for (offset, target) in row.iter_mut().enumerate() {
                        *target =
                            i32::try_from(crate::core::market_price(item, minimum + offset as i32))
                                .map_err(|_| "policy price exceeds int32")?;
                    }
                    Ok::<(), &str>(())
                })
        })
        .map_err(PyValueError::new_err)?;
        Ok(Array2::from_shape_vec((PRODUCTS, width), values)
            .expect("quote table shape")
            .into_pyarray(py))
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
            false,
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

    /// The built-in reference agents' actions for the current state, without
    /// stepping, in the same row order as `sample_and_step_into`. A row whose
    /// code is 0 comes back as an all-PASS action: this path never samples.
    /// Exactly one call per step is required for the scripted v27 rows: that
    /// agent advances its weed repair as a side effect, and the reference is
    /// not idempotent within a step either -- a second call at the same step
    /// sees age zero and abandons the repair it had just begun.
    fn builtin_actions<'py>(
        &mut self,
        py: Python<'py>,
        builtin_agents: PyReadonlyArray1<'py, u8>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let rows = self.games.len() * PLAYERS;
        ensure_shape(builtin_agents.shape(), &[rows], "builtin_agents")?;
        if !builtin_agents.is_c_contiguous() {
            return Err(PyValueError::new_err("builtin_agents must be C-contiguous"));
        }
        let codes = builtin_agents.as_slice()?;
        validate_builtin_agents(codes)?;
        let mut units = Vec::with_capacity(rows * MAX_UNITS);
        let mut kinds = Vec::with_capacity(rows * MAX_MARKET_ORDERS);
        let mut quantities = Vec::with_capacity(rows * MAX_MARKET_ORDERS);
        let games = &self.games;
        let v27_states = &mut self.v27_states;
        for (row, &code) in codes.iter().enumerate() {
            let game = &games[row / PLAYERS];
            let player = row % PLAYERS;
            let action = match BuiltinAgent::from_code(code).expect("codes are validated above") {
                Some(agent) => game.builtin_action(
                    player,
                    agent,
                    &mut builtin_rng(game, player),
                    &mut v27_states[row],
                ),
                None => CompactAction::default(),
            };
            units.extend(action.units);
            kinds.extend(action.market_kinds);
            quantities.extend(action.market_quantities);
        }
        let output = PyDict::new(py);
        output.set_item(
            "unit_actions",
            Array2::from_shape_vec((rows, MAX_UNITS), units)
                .expect("built-in unit shape is internal")
                .into_pyarray(py),
        )?;
        for (name, values) in [("market_kinds", kinds), ("market_quantities", quantities)] {
            output.set_item(
                name,
                Array2::from_shape_vec((rows, MAX_MARKET_ORDERS), values)
                    .expect("built-in market shape is internal")
                    .into_pyarray(py),
            )?;
        }
        Ok(output)
    }

    /// Allocating convenience wrapper around `sample_and_step_into`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        unit_logits, market_kind_logits, market_quantity_context,
        quantity_kind_gate, quantity_values, quantity_bias, head_ids,
        unit_draws, market_kind_draws, market_quantity_draws,
        deterministic_rows, temperatures, builtin_agents
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
        builtin_agents: PyReadonlyArray1<'py, u8>,
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
            builtin_agents,
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
        deterministic_rows, temperatures, builtin_agents, output
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
        builtin_agents: PyReadonlyArray1<'py, u8>,
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
        ensure_shape(builtin_agents.shape(), &[rows], "builtin_agents")?;
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
        require_c_input!(builtin_agents, "builtin_agents");

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
        let builtin_agents = builtin_agents.as_slice()?;
        validate_builtin_agents(builtin_agents)?;
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
            let v27_states = &mut self.v27_states;
            py.detach(|| {
                sampled
                    .par_iter_mut()
                    .zip(v27_states.par_iter_mut())
                    .enumerate()
                    .for_each(|(row, (output, v27_state))| {
                        let game = &games[row / PLAYERS];
                        let player = row % PLAYERS;
                        // A nonzero code only ever lands on a frozen opponent's
                        // seat. Nothing here can see which row belongs to the
                        // learner, so `collect_mixed_play_rust` in
                        // src/kaggriculture/rollout.py is where that invariant
                        // is enforced.
                        let builtin = BuiltinAgent::from_code(builtin_agents[row])
                            .expect("codes are validated above");
                        if let Some(agent) = builtin {
                            let action = game.builtin_action(
                                player,
                                agent,
                                &mut builtin_rng(game, player),
                                v27_state,
                            );
                            output.masks = game.factor_masks(player, &action);
                            output.action = action;
                            // The row's action never passed through the network,
                            // so it carries no policy density to report.
                            output.unit_logprobs.fill(0.0);
                            output.market_kind_logprobs.fill(0.0);
                            output.market_quantity_logprobs.fill(0.0);
                            output.unit_entropies.fill(0.0);
                            output.market_kind_entropies.fill(0.0);
                            output.market_quantity_entropies.fill(0.0);
                            output.mean_entropy = 0.0;
                            return;
                        }
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
        py.detach(|| {
            fill_sample_step_output(
                &self.games,
                &self.sampled_scratch,
                &self.results_scratch,
                &mut self.potential_cache,
                &mut output_slices,
            );
        });
        Ok(())
    }

    /// Apply GPU-produced categorical random utilities under exact engine
    /// legality, then advance every game.
    ///
    /// Unit and market-kind utilities already contain temperature scaling and
    /// Gumbel noise. Quantity logits remain low rank until the selected kind is
    /// known. Learned-row policy statistics stay zero for GPU reconstruction.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        unit_utilities, market_kind_utilities, market_quantity_context,
        quantity_kind_gate, quantity_values, quantity_bias, head_ids,
        market_quantity_draws, deterministic_rows, temperatures,
        builtin_agents, output
    ))]
    fn select_and_step_into<'py>(
        &mut self,
        py: Python<'py>,
        unit_utilities: PyReadonlyArray3<'py, f32>,
        market_kind_utilities: PyReadonlyArray3<'py, f32>,
        market_quantity_context: PyReadonlyArray3<'py, f32>,
        quantity_kind_gate: PyReadonlyArray3<'py, f32>,
        quantity_values: PyReadonlyArray3<'py, f32>,
        quantity_bias: PyReadonlyArray3<'py, f32>,
        head_ids: PyReadonlyArray1<'py, u16>,
        market_quantity_draws: PyReadonlyArray2<'py, f32>,
        deterministic_rows: PyReadonlyArray1<'py, bool>,
        temperatures: PyReadonlyArray1<'py, f32>,
        builtin_agents: PyReadonlyArray1<'py, u8>,
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
            unit_utilities.shape(),
            &[rows, MAX_UNITS, UNIT_ACTIONS],
            "unit_utilities",
        )?;
        ensure_shape(
            market_kind_utilities.shape(),
            &[rows, MAX_MARKET_ORDERS, MARKET_KINDS],
            "market_kind_utilities",
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
        ensure_shape(
            market_quantity_draws.shape(),
            &[rows, MAX_MARKET_ORDERS],
            "market_quantity_draws",
        )?;
        ensure_shape(deterministic_rows.shape(), &[rows], "deterministic_rows")?;
        ensure_shape(temperatures.shape(), &[rows], "temperatures")?;
        ensure_shape(builtin_agents.shape(), &[rows], "builtin_agents")?;

        require_c_input!(unit_utilities, "unit_utilities");
        require_c_input!(market_kind_utilities, "market_kind_utilities");
        require_c_input!(market_quantity_context, "market_quantity_context");
        require_c_input!(quantity_kind_gate, "quantity_kind_gate");
        require_c_input!(quantity_values, "quantity_values");
        require_c_input!(quantity_bias, "quantity_bias");
        require_c_input!(head_ids, "head_ids");
        require_c_input!(market_quantity_draws, "market_quantity_draws");
        require_c_input!(deterministic_rows, "deterministic_rows");
        require_c_input!(temperatures, "temperatures");
        require_c_input!(builtin_agents, "builtin_agents");

        let unit_utilities = unit_utilities.as_slice()?;
        let kind_utilities = market_kind_utilities.as_slice()?;
        let quantity_context = market_quantity_context.as_slice()?;
        let kind_gate = quantity_kind_gate.as_slice()?;
        let quantity_values = quantity_values.as_slice()?;
        let quantity_bias = quantity_bias.as_slice()?;
        let head_ids = head_ids.as_slice()?;
        let quantity_draws = market_quantity_draws.as_slice()?;
        let deterministic_rows = deterministic_rows.as_slice()?;
        let temperatures = temperatures.as_slice()?;
        let builtin_agents = builtin_agents.as_slice()?;
        validate_builtin_agents(builtin_agents)?;
        if head_ids.iter().any(|&head| usize::from(head) >= heads) {
            return Err(PyValueError::new_err(
                "head_ids contains an out-of-range head",
            ));
        }
        if quantity_draws
            .iter()
            .any(|&draw| !draw.is_finite() || !(0.0..1.0).contains(&draw))
        {
            return Err(PyValueError::new_err(
                "market_quantity_draws must contain finite values in [0, 1)",
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
        // These scans cover ~625K floats per step with the GIL held. A
        // short-circuiting `any` is scalar; folding a bit-or of the exponent
        // bits vectorizes, and the two large policy tables split across rayon.
        for (name, values) in [
            ("unit_utilities", unit_utilities),
            ("market_kind_utilities", kind_utilities),
            ("market_quantity_context", quantity_context),
            ("quantity_kind_gate", kind_gate),
            ("quantity_values", quantity_values),
            ("quantity_bias", quantity_bias),
        ] {
            if !all_finite(values) {
                return Err(PyValueError::new_err(format!(
                    "{name} must contain finite values"
                )));
            }
        }

        let mut output_arrays = SampleOutputArrays::new(output, rows, self.games.len())?;
        let mut output_slices = output_arrays.slices()?;
        let games = &mut self.games;
        let sampled = &mut self.sampled_scratch;
        let results = &mut self.results_scratch;
        let potential_cache = &mut self.potential_cache;
        let v27_states = &mut self.v27_states;
        py.detach(|| {
            games
                .par_iter_mut()
                .zip(sampled.par_chunks_mut(PLAYERS))
                .zip(v27_states.par_chunks_mut(PLAYERS))
                .zip(results.par_iter_mut())
                .enumerate()
                .for_each(|(game_index, (((game, sampled_rows), v27_rows), result))| {
                    let first_row = game_index * PLAYERS;
                    for player in 0..PLAYERS {
                        let row = first_row + player;
                        let sampled_row = &mut sampled_rows[player];
                        let builtin = BuiltinAgent::from_code(builtin_agents[row])
                            .expect("codes are validated above");
                        if let Some(agent) = builtin {
                            let action = game.builtin_action(
                                player,
                                agent,
                                &mut builtin_rng(game, player),
                                &mut v27_rows[player],
                            );
                            sampled_row.masks = game.factor_masks(player, &action);
                            sampled_row.action = action;
                            sampled_row.unit_logprobs.fill(0.0);
                            sampled_row.market_kind_logprobs.fill(0.0);
                            sampled_row.market_quantity_logprobs.fill(0.0);
                            sampled_row.unit_entropies.fill(0.0);
                            sampled_row.market_kind_entropies.fill(0.0);
                            sampled_row.market_quantity_entropies.fill(0.0);
                            sampled_row.mean_entropy = 0.0;
                            continue;
                        }

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
                        let quantity_draw_offset = row * MAX_MARKET_ORDERS;
                        *sampled_row = game.select_factors(
                            player,
                            &unit_utilities[unit_offset..unit_offset + MAX_UNITS * UNIT_ACTIONS],
                            &kind_utilities
                                [kind_offset..kind_offset + MAX_MARKET_ORDERS * MARKET_KINDS],
                            &quantity_context
                                [context_offset..context_offset + MAX_MARKET_ORDERS * rank],
                            &head,
                            &quantity_draws
                                [quantity_draw_offset..quantity_draw_offset + MAX_MARKET_ORDERS],
                            deterministic_rows[row],
                            temperatures[row],
                        );
                    }
                    *result = game.step(&[sampled_rows[0].action, sampled_rows[1].action]);
                });
            fill_sample_step_output(games, sampled, results, potential_cache, &mut output_slices);
        });
        Ok(())
    }

    /// Advance already-selected causal factors without resampling any policy row.
    /// Built-ins are evaluated exactly once at this boundary. Policy densities
    /// remain zero here: the device decoder supplies them with its chosen prefix.
    fn step_factors_into<'py>(
        &mut self,
        py: Python<'py>,
        unit_actions: PyReadonlyArray2<'py, u8>,
        market_kinds: PyReadonlyArray2<'py, u8>,
        market_quantities: PyReadonlyArray2<'py, u8>,
        builtin_agents: PyReadonlyArray1<'py, u8>,
        output: &Bound<'py, PyDict>,
    ) -> PyResult<()> {
        let rows = self.games.len() * PLAYERS;
        ensure_shape(unit_actions.shape(), &[rows, MAX_UNITS], "unit_actions")?;
        ensure_shape(
            market_kinds.shape(),
            &[rows, MAX_MARKET_ORDERS],
            "market_kinds",
        )?;
        ensure_shape(
            market_quantities.shape(),
            &[rows, MAX_MARKET_ORDERS],
            "market_quantities",
        )?;
        ensure_shape(builtin_agents.shape(), &[rows], "builtin_agents")?;
        if !unit_actions.is_c_contiguous()
            || !market_kinds.is_c_contiguous()
            || !market_quantities.is_c_contiguous()
            || !builtin_agents.is_c_contiguous()
        {
            return Err(PyValueError::new_err("factor inputs must be C-contiguous"));
        }
        let units = unit_actions.as_slice()?;
        let kinds = market_kinds.as_slice()?;
        let quantities = market_quantities.as_slice()?;
        let codes = builtin_agents.as_slice()?;
        validate_builtin_agents(codes)?;
        if units.iter().any(|&a| usize::from(a) >= UNIT_ACTIONS)
            || kinds.iter().any(|&a| usize::from(a) >= MARKET_KINDS)
            || quantities
                .iter()
                .any(|&a| usize::from(a) >= MARKET_QUANTITIES)
        {
            return Err(PyValueError::new_err(
                "factor index outside categorical support",
            ));
        }
        let mut output_arrays = SampleOutputArrays::new(output, rows, self.games.len())?;
        let mut output_slices = output_arrays.slices()?;
        let games = &mut self.games;
        let sampled = &mut self.sampled_scratch;
        let results = &mut self.results_scratch;
        let potential_cache = &mut self.potential_cache;
        let v27_states = &mut self.v27_states;
        py.detach(|| {
            games
                .par_iter_mut()
                .zip(sampled.par_chunks_mut(PLAYERS))
                .zip(v27_states.par_chunks_mut(PLAYERS))
                .zip(results.par_iter_mut())
                .enumerate()
                .for_each(|(game_index, (((game, sampled_rows), v27_rows), result))| {
                    for player in 0..PLAYERS {
                        let row = game_index * PLAYERS + player;
                        let action = if let Some(agent) =
                            BuiltinAgent::from_code(codes[row]).expect("codes validated above")
                        {
                            game.builtin_action(
                                player,
                                agent,
                                &mut builtin_rng(game, player),
                                &mut v27_rows[player],
                            )
                        } else {
                            let mut action = CompactAction::default();
                            action
                                .units
                                .copy_from_slice(&units[row * MAX_UNITS..(row + 1) * MAX_UNITS]);
                            action.market_kinds.copy_from_slice(
                                &kinds[row * MAX_MARKET_ORDERS..(row + 1) * MAX_MARKET_ORDERS],
                            );
                            action.market_quantities.copy_from_slice(
                                &quantities[row * MAX_MARKET_ORDERS..(row + 1) * MAX_MARKET_ORDERS],
                            );
                            action
                        };
                        let sampled_row = &mut sampled_rows[player];
                        sampled_row.masks = game.factor_masks(player, &action);
                        sampled_row.action = action;
                        sampled_row.unit_logprobs.fill(0.0);
                        sampled_row.market_kind_logprobs.fill(0.0);
                        sampled_row.market_quantity_logprobs.fill(0.0);
                        sampled_row.unit_entropies.fill(0.0);
                        sampled_row.market_kind_entropies.fill(0.0);
                        sampled_row.market_quantity_entropies.fill(0.0);
                        sampled_row.mean_entropy = 0.0;
                    }
                    *result = game.step(&[sampled_rows[0].action, sampled_rows[1].action]);
                });
            fill_sample_step_output(games, sampled, results, potential_cache, &mut output_slices);
        });
        Ok(())
    }

    /// Advance every game by one supplied factor row.
    ///
    /// `external` marks the rows as an outside agent's submitted dict rather than
    /// our policy's masked sample, which is what the parity harnesses replay: the
    /// interpreter clamps a partial pickup and drops over-demanded plants, while
    /// our own factor space excludes both by construction.
    #[pyo3(signature = (unit_actions, market_kinds, market_quantities, external = false))]
    fn step_factors<'py>(
        &mut self,
        py: Python<'py>,
        unit_actions: PyReadonlyArray3<'py, u8>,
        market_kinds: PyReadonlyArray3<'py, u8>,
        market_quantities: PyReadonlyArray3<'py, u8>,
        external: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let compact = extract_compact_actions(
            self.games.len(),
            unit_actions,
            market_kinds,
            market_quantities,
            external,
        )?;
        let previous_potentials = self.potential_cache.clone();
        let results = py.detach(|| {
            self.games
                .par_iter_mut()
                .zip(compact.par_iter())
                .map(|(game, actions)| game.step(actions))
                .collect::<Vec<_>>()
        });
        build_step_output(
            py,
            &self.games,
            results,
            previous_potentials,
            &mut self.potential_cache,
        )
    }

    /// Advance every game with variable-length official submitted-dict unit rows.
    ///
    /// Unit rows are ragged `[game][player][submitted unit command]` Python
    /// lists. They may omit live hands or include commands for nonexistent hands,
    /// matching the official dict interpreter. Market factors retain the fixed
    /// policy tensor shape because the engine's unit count, not its market queue,
    /// is the dimension that can exceed the model.
    #[pyo3(signature = (unit_actions, market_kinds, market_quantities))]
    fn step_submitted<'py>(
        &mut self,
        py: Python<'py>,
        unit_actions: Vec<Vec<Vec<u8>>>,
        market_kinds: PyReadonlyArray3<'py, u8>,
        market_quantities: PyReadonlyArray3<'py, u8>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (submitted_units, market_actions) =
            extract_submitted_actions(&self.games, unit_actions, market_kinds, market_quantities)?;
        let previous_potentials = self.potential_cache.clone();
        let results = py.detach(|| {
            self.games
                .par_iter_mut()
                .zip(submitted_units.par_iter())
                .zip(market_actions.par_iter())
                .map(|((game, units), market)| {
                    game.step_submitted(market, [units[0].as_slice(), units[1].as_slice()])
                })
                .collect::<Vec<_>>()
        });
        build_step_output(
            py,
            &self.games,
            results,
            previous_potentials,
            &mut self.potential_cache,
        )
    }
}

fn build_step_output<'py>(
    py: Python<'py>,
    games: &[Game],
    results: Vec<StepResult>,
    previous_potentials: Vec<f32>,
    potential_cache: &mut [f32],
) -> PyResult<Bound<'py, PyDict>> {
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
        Array2::from_shape_vec((games.len(), PLAYERS), rewards)
            .expect("step reward shape is internal")
            .into_pyarray(py),
    )?;
    output.set_item(
        "final_money",
        Array2::from_shape_vec((games.len(), PLAYERS), money)
            .expect("step money shape is internal")
            .into_pyarray(py),
    )?;
    output.set_item("dones", dones.into_pyarray(py))?;
    let post_potentials: Vec<f32> = games.iter().map(Game::post_step_potential).collect();
    potential_cache.copy_from_slice(&post_potentials);
    let terminal_utilities: Vec<f32> = games
        .iter()
        .map(|game| {
            if game.done {
                game.terminal_pair_utility()
            } else {
                0.0
            }
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
        "terminal_utilities",
        Array1::from_vec(terminal_utilities).into_pyarray(py),
    )?;
    Ok(output)
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
        "animals",
        PyArray3::<f16>::zeros(py, [rows, ANIMALS, ANIMAL_TOKEN_FIELDS], false),
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
    let mut animals = output_array!(
        "animals",
        PyArray3<f16>,
        [rows, ANIMALS, ANIMAL_TOKEN_FIELDS]
    );
    let mut crops = output_array!("crops", PyArray3<f16>, [rows, CROPS, CROP_TOKEN_FIELDS]);
    let mut farms = output_array!("farms", PyArray3<f16>, [rows, PLAYERS, FARM_TOKEN_FIELDS]);
    let mut town = output_array!("town", PyArray2<f16>, [rows, TOWN_TOKEN_FIELDS]);

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
    let animals = animals
        .as_slice_mut()
        .map_err(|_| non_contiguous("animals"))?;
    let crops = crops.as_slice_mut().map_err(|_| non_contiguous("crops"))?;
    let farms = farms.as_slice_mut().map_err(|_| non_contiguous("farms"))?;
    let town = town.as_slice_mut().map_err(|_| non_contiguous("town"))?;

    const TILE_CATEGORICAL_VALUES: usize = TILE_TOKENS * TILE_CATEGORICAL;
    const TILE_CONTINUOUS_VALUES: usize = TILE_TOKENS * TILE_CONTINUOUS;
    const UNIT_CATEGORICAL_VALUES: usize = MAX_UNITS * UNIT_CATEGORICAL;
    const UNIT_CONTINUOUS_VALUES: usize = MAX_UNITS * UNIT_CONTINUOUS;
    const UNIT_GATHER_VALUES: usize = MAX_UNITS * UNIT_GATHERS;
    const PRODUCT_VALUES: usize = PRODUCTS * PRODUCT_TOKEN_FIELDS;
    const ANIMAL_VALUES: usize = ANIMALS * ANIMAL_TOKEN_FIELDS;
    const CROP_VALUES: usize = CROPS * CROP_TOKEN_FIELDS;
    const FARM_VALUES: usize = PLAYERS * FARM_TOKEN_FIELDS;
    py.detach(|| {
        tile_categorical
            .par_chunks_mut(PLAYERS * TILE_CATEGORICAL_VALUES)
            .zip(tile_continuous.par_chunks_mut(PLAYERS * TILE_CONTINUOUS_VALUES))
            .zip(unit_categorical.par_chunks_mut(PLAYERS * UNIT_CATEGORICAL_VALUES))
            .zip(unit_continuous.par_chunks_mut(PLAYERS * UNIT_CONTINUOUS_VALUES))
            .zip(unit_active.par_chunks_mut(PLAYERS * MAX_UNITS))
            .zip(unit_tile_gather.par_chunks_mut(PLAYERS * UNIT_GATHER_VALUES))
            .zip(unit_tile_gather_valid.par_chunks_mut(PLAYERS * UNIT_GATHER_VALUES))
            .zip(products.par_chunks_mut(PLAYERS * PRODUCT_VALUES))
            .zip(crops.par_chunks_mut(PLAYERS * CROP_VALUES))
            .zip(farms.par_chunks_mut(PLAYERS * FARM_VALUES))
            .zip(
                town.par_chunks_mut(PLAYERS * TOWN_TOKEN_FIELDS)
                    .zip(animals.par_chunks_mut(PLAYERS * ANIMAL_VALUES)),
            )
            .enumerate()
            .for_each(
                |(
                    game,
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
                        (town, animals),
                    ),
                )| {
                    encode_game_structured(
                        &games[game],
                        tile_categorical,
                        tile_continuous,
                        unit_categorical,
                        unit_continuous,
                        unit_active,
                        unit_tile_gather,
                        unit_tile_gather_valid,
                        products,
                        animals,
                        crops,
                        farms,
                        town,
                    );
                },
            );
    });
    Ok(())
}

/// Fill game-major, seat-minor rows, retaining both private views for the critic.
#[allow(clippy::too_many_arguments)]
fn encode_game_structured(
    game: &Game,
    tile_categorical: &mut [i8],
    tile_continuous: &mut [f16],
    unit_categorical: &mut [i8],
    unit_continuous: &mut [f16],
    unit_active: &mut [bool],
    unit_tile_gather: &mut [i8],
    unit_tile_gather_valid: &mut [bool],
    products: &mut [f16],
    animals: &mut [f16],
    crops: &mut [f16],
    farms: &mut [f16],
    town: &mut [f16],
) {
    game.encode_pair_structured_tiles(tile_categorical, tile_continuous, f16::from_f32);
    let mut unit_continuous_f32 = [0.0f32; MAX_UNITS * UNIT_CONTINUOUS];
    let mut products_f32 = [0.0f32; PRODUCTS * PRODUCT_TOKEN_FIELDS];
    let mut animals_f32 = [0.0f32; ANIMALS * ANIMAL_TOKEN_FIELDS];
    let mut crops_f32 = [0.0f32; CROPS * CROP_TOKEN_FIELDS];
    let mut farms_f32 = [0.0f32; PLAYERS * FARM_TOKEN_FIELDS];
    let mut town_f32 = [0.0f32; TOWN_TOKEN_FIELDS];
    for player in 0..PLAYERS {
        game.encode_player_structured_state(
            player,
            &mut unit_categorical[player * MAX_UNITS * UNIT_CATEGORICAL
                ..(player + 1) * MAX_UNITS * UNIT_CATEGORICAL],
            &mut unit_continuous_f32,
            &mut unit_active[player * MAX_UNITS..(player + 1) * MAX_UNITS],
            &mut unit_tile_gather
                [player * MAX_UNITS * UNIT_GATHERS..(player + 1) * MAX_UNITS * UNIT_GATHERS],
            &mut unit_tile_gather_valid
                [player * MAX_UNITS * UNIT_GATHERS..(player + 1) * MAX_UNITS * UNIT_GATHERS],
            &mut products_f32,
            &mut animals_f32,
            &mut crops_f32,
            &mut farms_f32,
            &mut town_f32,
        );
        for (output, values) in [
            (&mut *unit_continuous, unit_continuous_f32.as_slice()),
            (&mut *products, products_f32.as_slice()),
            (&mut *animals, animals_f32.as_slice()),
            (&mut *crops, crops_f32.as_slice()),
            (&mut *farms, farms_f32.as_slice()),
            (&mut *town, town_f32.as_slice()),
        ] {
            for (target, &value) in output[player * values.len()..(player + 1) * values.len()]
                .iter_mut()
                .zip(values)
            {
                *target = f16::from_f32(value);
            }
        }
    }
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
    for name in ["rewards", "final_money"] {
        output.set_item(name, PyArray2::<f32>::zeros(py, [batch, PLAYERS], false))?;
    }
    output.set_item("dones", PyArray1::<bool>::zeros(py, batch, false))?;
    for name in ["previous_potentials", "potentials", "terminal_utilities"] {
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
    utilities: PyReadwriteArray1<'py, f32>,
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
            utilities: output_array!("terminal_utilities", PyArray1<f32>, [batch]),
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
            utilities,
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
            utilities: utilities
                .as_slice_mut()
                .map_err(|_| non_contiguous("terminal_utilities"))?,
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
    utilities: &'a mut [f32],
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
    });
    Ok(())
}

/// Whether every value is finite, as a branch-free fold that LLVM vectorizes.
///
/// A float is non-finite exactly when its exponent bits are all set, so the
/// bit-and of every value's exponent field saturates at the mask iff some
/// value is infinite or NaN. Slices past a few hundred KiB split across rayon
/// so the per-step policy tables do not serialize on one core.
fn all_finite(values: &[f32]) -> bool {
    const EXPONENT: u32 = 0x7f80_0000;
    const PARALLEL_CHUNK: usize = 1 << 16;
    fn finite_chunk(chunk: &[f32]) -> bool {
        chunk.iter().fold(0u32, |seen, value| {
            seen | ((value.to_bits() & EXPONENT) == EXPONENT) as u32
        }) == 0
    }
    if values.len() <= PARALLEL_CHUNK {
        finite_chunk(values)
    } else {
        values.par_chunks(PARALLEL_CHUNK).all(finite_chunk)
    }
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

fn validate_builtin_agents(codes: &[u8]) -> PyResult<()> {
    if let Some(&code) = codes
        .iter()
        .find(|&&code| BuiltinAgent::from_code(code).is_err())
    {
        return Err(PyValueError::new_err(format!(
            "builtin_agents contains unknown agent code {code}, expected 0..=4"
        )));
    }
    Ok(())
}

/// Deterministic draw stream for the built-in `random` agent.
///
/// The reference agent uses an unseeded `random.Random()`, so there is no draw
/// sequence to reproduce and seeding costs no fidelity; deriving the stream
/// from the game's own seed, step and seat instead makes a league rollout that
/// fields the random opponent exactly replayable. The salt keeps it clear of
/// the end-of-day stream, which mixes the same seed with the day index.
fn builtin_rng(game: &Game, player: usize) -> PyRandom {
    const SALT: u64 = 0x9e37_79b9_7f4a_7c15;
    PyRandom::seed_u64(
        SALT ^ (game.seed * 1_000_003) ^ (u64::from(game.step) * PLAYERS as u64 + player as u64),
    )
}

fn validate_submitted_unit_actions(
    games: &[Game],
    unit_actions: &[Vec<Vec<u8>>],
) -> Result<(), String> {
    if unit_actions.len() != games.len() {
        return Err(format!(
            "unit_actions has {} games, expected {}",
            unit_actions.len(),
            games.len()
        ));
    }
    for (game_index, players) in unit_actions.iter().enumerate() {
        if players.len() != PLAYERS {
            return Err(format!(
                "unit_actions[{game_index}] has {} player rows, expected {PLAYERS}",
                players.len()
            ));
        }
        for (player, actions) in players.iter().enumerate() {
            if let Some((unit, action)) = actions
                .iter()
                .copied()
                .enumerate()
                .find(|(_, action)| usize::from(*action) >= UNIT_ACTIONS)
            {
                return Err(format!(
                    "unit_actions[{game_index}][{player}][{unit}] is {action}, expected 0..{}",
                    UNIT_ACTIONS - 1
                ));
            }
        }
    }
    Ok(())
}

type SubmittedActions = (Vec<[Vec<u8>; PLAYERS]>, Vec<[CompactAction; PLAYERS]>);

fn extract_submitted_actions(
    games: &[Game],
    unit_actions: Vec<Vec<Vec<u8>>>,
    market_kinds: PyReadonlyArray3<'_, u8>,
    market_quantities: PyReadonlyArray3<'_, u8>,
) -> PyResult<SubmittedActions> {
    validate_submitted_unit_actions(games, &unit_actions).map_err(PyValueError::new_err)?;
    let expected_market = [games.len(), PLAYERS, MAX_MARKET_ORDERS];
    if market_kinds.shape() != expected_market || market_quantities.shape() != expected_market {
        return Err(PyValueError::new_err(format!(
            "market factor shapes {:?}/{:?}, expected {expected_market:?}",
            market_kinds.shape(),
            market_quantities.shape()
        )));
    }

    let submitted = unit_actions
        .into_iter()
        .map(|players| {
            players
                .try_into()
                .expect("submitted unit rows were validated above")
        })
        .collect();

    let kinds = market_kinds.as_array();
    let quantities = market_quantities.as_array();
    let mut market_actions = Vec::with_capacity(games.len());
    for game in 0..games.len() {
        let mut rows = [CompactAction::default(); PLAYERS];
        for player in 0..PLAYERS {
            rows[player].external = true;
            for slot in 0..MAX_MARKET_ORDERS {
                let kind = kinds[[game, player, slot]];
                let quantity = quantities[[game, player, slot]];
                if usize::from(kind) >= MARKET_KINDS {
                    return Err(PyValueError::new_err(format!(
                        "market_kinds[{game}][{player}][{slot}] is {kind}, expected 0..{}",
                        MARKET_KINDS - 1
                    )));
                }
                if usize::from(quantity) >= MARKET_QUANTITIES {
                    return Err(PyValueError::new_err(format!(
                        "market_quantities[{game}][{player}][{slot}] is {quantity}, expected 0..{}",
                        MARKET_QUANTITIES - 1
                    )));
                }
                rows[player].market_kinds[slot] = kind;
                rows[player].market_quantities[slot] = quantity;
            }
        }
        market_actions.push(rows);
    }
    Ok((submitted, market_actions))
}

fn extract_compact_actions(
    games: usize,
    unit_actions: PyReadonlyArray3<'_, u8>,
    market_kinds: PyReadonlyArray3<'_, u8>,
    market_quantities: PyReadonlyArray3<'_, u8>,
    external: bool,
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
                external,
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
        utilities,
    } = output;

    sampled
        .par_iter()
        .zip(unit_actions.par_chunks_mut(MAX_UNITS))
        .zip(market_kinds.par_chunks_mut(MAX_MARKET_ORDERS))
        .zip(market_quantities.par_chunks_mut(MAX_MARKET_ORDERS))
        .for_each(|(((row, units), kinds), quantities)| {
            units.copy_from_slice(&row.action.units);
            kinds.copy_from_slice(&row.action.market_kinds);
            quantities.copy_from_slice(&row.action.market_quantities);
        });
    sampled
        .par_iter()
        .zip(unit_masks.par_chunks_mut(MAX_UNITS * UNIT_ACTIONS))
        .zip(market_kind_masks.par_chunks_mut(MAX_MARKET_ORDERS * MARKET_KINDS))
        .zip(market_quantity_masks.par_chunks_mut(MAX_MARKET_ORDERS * MARKET_QUANTITIES))
        .for_each(|(((row, unit), kinds), quantities)| {
            unit.copy_from_slice(&row.masks.unit);
            kinds.copy_from_slice(&row.masks.market_kind);
            quantities.copy_from_slice(&row.masks.market_quantity);
        });
    sampled
        .par_iter()
        .zip(unit_active.par_chunks_mut(MAX_UNITS))
        .zip(market_active.par_chunks_mut(MAX_MARKET_ORDERS))
        .zip(market_quantity_active.par_chunks_mut(MAX_MARKET_ORDERS))
        .for_each(|(((row, units), markets), quantities)| {
            units.copy_from_slice(&row.masks.unit_active);
            markets.copy_from_slice(&row.masks.market_active);
            quantities.copy_from_slice(&row.masks.market_quantity_active);
        });
    sampled
        .par_iter()
        .zip(unit_logprobs.par_chunks_mut(MAX_UNITS))
        .zip(market_kind_logprobs.par_chunks_mut(MAX_MARKET_ORDERS))
        .zip(market_quantity_logprobs.par_chunks_mut(MAX_MARKET_ORDERS))
        .zip(entropy.par_iter_mut())
        .for_each(|((((row, units), kinds), quantities), entropy)| {
            units.copy_from_slice(&row.unit_logprobs);
            kinds.copy_from_slice(&row.market_kind_logprobs);
            quantities.copy_from_slice(&row.market_quantity_logprobs);
            *entropy = row.mean_entropy;
        });

    games
        .par_iter()
        .zip(results.par_iter())
        .zip(potential_cache.par_iter_mut())
        .zip(rewards.par_chunks_mut(PLAYERS))
        .zip(money.par_chunks_mut(PLAYERS))
        .zip(dones.par_iter_mut())
        .zip(previous.par_iter_mut())
        .zip(potentials.par_iter_mut())
        .zip(utilities.par_iter_mut())
        .for_each(
            |(
                (((((((game, result), cached), rewards), money), done), previous), potential),
                utility,
            )| {
                rewards.copy_from_slice(&result.rewards);
                money.copy_from_slice(&result.money);
                *done = result.done;
                *previous = *cached;
                let post = game.post_step_potential();
                *potential = post;
                *utility = if result.done {
                    game.terminal_pair_utility()
                } else {
                    0.0
                };
                *cached = post;
            },
        );
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<BatchEnv>()?;
    module.add("OBSERVATION_SCHEMA_VERSION", OBSERVATION_SCHEMA_VERSION)?;
    module.add("POLICY_LEDGER_SCHEMA_VERSION", POLICY_LEDGER_SCHEMA_VERSION)?;
    module.add("POLICY_LEDGER_WIDTH", POLICY_LEDGER_WIDTH)?;
    module.add("MAX_UNITS", MAX_UNITS)?;
    module.add("MAX_MARKET_ORDERS", MAX_MARKET_ORDERS)?;
    module.add("N_UNIT_ACTIONS", UNIT_ACTIONS)?;
    module.add("N_MARKET_KINDS", MARKET_KINDS)?;
    module.add("N_QUANTITIES", MARKET_QUANTITIES)?;
    // The scripted v27 built-in replays a table compiled from one exact agent
    // file. Publishing that file's digest lets a parity audit prove it is
    // comparing the native port against the very bytes it was built from,
    // rather than against a copy that has since moved.
    module.add("V27_SOURCE_SHA256", V27_SOURCE_SHA256)?;
    module.add("V27_SOURCE_NAME", V27_SOURCE_NAME)?;
    module.add("V27_STEPS", V27_STEPS)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_structured_rows_match_single_seats_and_preserve_private_stock() {
        use crate::core::{PRIVATE_ITEMS, Position, TILE_COUNT, Tile, TileKind};

        let mut game = Game::new(17, GameConfig::default());
        for player in 0..PLAYERS {
            game.farms[player].money += player as i64 * 1234;
            game.farms[player].positions[0] = Position(player as u8 * 9, 0);
            game.farms[player].positions.push(Position(9, 9));
            game.privates[player].inventories.push([0; PRIVATE_ITEMS]);
            game.privates[player]
                .inventory_order
                .push([u8::MAX; PRIVATE_ITEMS]);
            game.privates[player].shed[0] = 11 + player as u16;
            game.privates[player].shed[PRODUCTS] = 3 + player as u16;
            game.privates[player].seeds[0] = 5 + player as u16;
            game.privates[player].inventories[0][0] = 7 + player as u16;
            game.privates[player].inventories[0][PRODUCTS + 1] = 2 + player as u16;
            game.privates[player].inventory_order[0][0] = (PRODUCTS + 1) as u8;
            game.privates[player].inventory_order[0][1] = 0;
            for (token, tile) in game.farms[player].tiles.iter_mut().enumerate() {
                let kind = match (token + player) % 6 {
                    0 => TileKind::Locked,
                    1 => TileKind::Empty,
                    2 => TileKind::Weed,
                    3 => TileKind::Plant,
                    4 => TileKind::Coop,
                    _ => TileKind::Pasture,
                };
                *tile = Tile {
                    kind,
                    species: if kind == TileKind::Plant {
                        (token % CROPS) as u8
                    } else if kind == TileKind::Pasture {
                        1 + (token % 2) as u8
                    } else {
                        0
                    },
                    has_animal: token % 3 != 0,
                    origin_day: player as u16,
                    yield_units: (token % 5) as u8,
                    consecutive_unmet: (token % 4) as u8,
                    watered_or_fed: token % 2 == 0,
                    cared_today: token % 3 == 0,
                    fertilizer_available: token % 4 == 0,
                    pending_care_bonus: (token % 6) as u8,
                    max_lifespan_step: if token % 2 == 0 { 240 } else { -1 },
                    fertilized_until_day: 10,
                };
            }
        }

        macro_rules! compare_rows {
            ($check:block; $(($paired:ident, $single:ident, $staged:ty, $native:ty, $width:expr, $convert:expr)),+ $(,)?) => {
                $(
                    let mut $paired = vec![<$staged>::default(); PLAYERS * $width];
                    let mut $single = vec![<$native>::default(); $width];
                )+
                // Reuse the same output buffers across a reset as the rollout does.
                for step in [0, 250, 719] {
                    game.step = step;
                    encode_game_structured(&game, $(&mut $paired),+);
                    for player in 0..PLAYERS {
                        game.encode_player_structured(player, $(&mut $single),+);
                        $(
                            let expected: Vec<$staged> =
                                $single.iter().copied().map($convert).collect();
                            assert_eq!(
                                &$paired[player * $width..(player + 1) * $width],
                                expected.as_slice(),
                                "{} differs at step {step}, seat {player}",
                                stringify!($paired),
                            );
                        )+
                    }
                }
                $check
                game = Game::new(23, GameConfig::default());
                encode_game_structured(&game, $(&mut $paired),+);
                for player in 0..PLAYERS {
                    game.encode_player_structured(player, $(&mut $single),+);
                    $(
                        let expected: Vec<$staged> =
                            $single.iter().copied().map($convert).collect();
                        assert_eq!(
                            &$paired[player * $width..(player + 1) * $width],
                            expected.as_slice(),
                            "{} retains stale values after reset, seat {player}",
                            stringify!($paired),
                        );
                    )+
                }
            };
        }
        compare_rows!(
            {
                assert_ne!(
                    &units[..MAX_UNITS * UNIT_CONTINUOUS],
                    &units[MAX_UNITS * UNIT_CONTINUOUS..],
                );
                assert_ne!(
                    &products[..PRODUCTS * PRODUCT_TOKEN_FIELDS],
                    &products[PRODUCTS * PRODUCT_TOKEN_FIELDS..],
                );
                assert_ne!(
                    &animals[..ANIMALS * ANIMAL_TOKEN_FIELDS],
                    &animals[ANIMALS * ANIMAL_TOKEN_FIELDS..],
                );
                assert_ne!(
                    &crops[..CROPS * CROP_TOKEN_FIELDS],
                    &crops[CROPS * CROP_TOKEN_FIELDS..],
                );
                for categorical in tiles.chunks_exact(TILE_TOKENS * TILE_CATEGORICAL) {
                    for (token, row) in categorical.chunks_exact(TILE_CATEGORICAL).enumerate() {
                        assert_eq!(row[2], i8::from(token >= TILE_COUNT));
                    }
                }
            };
            (tiles, single_tiles, i8, i8, TILE_TOKENS * TILE_CATEGORICAL, std::convert::identity),
            (tile_values, single_tile_values, f16, f32, TILE_TOKENS * TILE_CONTINUOUS, f16::from_f32),
            (unit_kinds, single_unit_kinds, i8, i8, MAX_UNITS * UNIT_CATEGORICAL, std::convert::identity),
            (units, single_units, f16, f32, MAX_UNITS * UNIT_CONTINUOUS, f16::from_f32),
            (active, single_active, bool, bool, MAX_UNITS, std::convert::identity),
            (gathers, single_gathers, i8, i8, MAX_UNITS * UNIT_GATHERS, std::convert::identity),
            (valid, single_valid, bool, bool, MAX_UNITS * UNIT_GATHERS, std::convert::identity),
            (products, single_products, f16, f32, PRODUCTS * PRODUCT_TOKEN_FIELDS, f16::from_f32),
            (animals, single_animals, f16, f32, ANIMALS * ANIMAL_TOKEN_FIELDS, f16::from_f32),
            (crops, single_crops, f16, f32, CROPS * CROP_TOKEN_FIELDS, f16::from_f32),
            (farms, single_farms, f16, f32, PLAYERS * FARM_TOKEN_FIELDS, f16::from_f32),
            (town, single_town, f16, f32, TOWN_TOKEN_FIELDS, f16::from_f32),
        );
    }

    #[test]
    fn submitted_unit_rows_accept_omitted_and_excess_hand_commands() {
        let game = Game::new(0, GameConfig::default());
        let actions = vec![vec![vec![], vec![0, 45]]];

        assert_eq!(validate_submitted_unit_actions(&[game], &actions), Ok(()));
    }

    #[test]
    fn submitted_unit_rows_reject_shape_and_action_range_errors() {
        let game = Game::new(0, GameConfig::default());
        assert!(validate_submitted_unit_actions(std::slice::from_ref(&game), &[]).is_err());
        assert!(
            validate_submitted_unit_actions(std::slice::from_ref(&game), &[vec![vec![0]]]).is_err()
        );
        assert!(
            validate_submitted_unit_actions(&[game], &[vec![vec![UNIT_ACTIONS as u8], vec![0]]],)
                .is_err()
        );
    }
}
