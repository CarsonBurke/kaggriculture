//! Exact, compact Kaggriculture simulator and contiguous PyO3 batch API.

mod core;
mod python;
mod rng;
mod v27_script;

pub use core::*;

use pyo3::prelude::*;

#[pymodule]
fn _kagg_env(module: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(module)
}
