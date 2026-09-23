from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def probe():
    path = Path(__file__).parents[1] / "scripts" / "probe_critic_source_attention.py"
    spec = importlib.util.spec_from_file_location("critic_source_attention_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_uniform_attention_distinguishes_mass_from_count_prior(probe):
    valid = np.ones((2, 260), dtype=bool)
    valid[:, -16:] = False
    probabilities = np.repeat((valid / valid.sum(axis=1, keepdims=True))[:, None], 4, axis=1)
    summary = probe.attention_summary(probabilities, valid)
    assert summary["groups"]["own_farm"]["attention_mass_mean"] == pytest.approx(100 / 244)
    assert summary["groups"]["own_farm"][
        "mass_over_count_prior_mean_when_present"
    ] == pytest.approx(1)
    assert summary["entropy_fraction_of_uniform_mean"] == pytest.approx(1)
    assert summary["groups"]["private_units"]["attention_per_valid_token_mean_when_present"] is None
    probabilities[:, :, -1] = 0.1
    with pytest.raises(ValueError, match="valid context"):
        probe.attention_summary(probabilities, valid)


@pytest.mark.parametrize("architecture,source_read", [("economic", True), ("entity", False)])
def test_probe_rejects_non_source_pool_layouts(probe, architecture, source_read):
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="entity critic"):
        probe.require_entity_source_pool(
            SimpleNamespace(critic_architecture=architecture, critic_source_read=source_read)
        )
