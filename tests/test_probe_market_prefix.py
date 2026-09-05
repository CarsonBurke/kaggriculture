from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_probe():
    path = Path(__file__).parents[1] / "scripts" / "probe_market_prefix.py"
    spec = importlib.util.spec_from_file_location("probe_market_prefix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return _load_probe()


def test_seed_split_is_whole_disjoint_and_deterministic(probe) -> None:
    # Duplicate seeds represent the two episode seats; neither may be divided.
    episode_seeds = [14, 10, 13, 11, 12, 10, 11, 12, 13, 14]

    first = probe.split_seed_sets(
        episode_seeds,
        holdout_seeds=4,
        validation_holdout_seeds=2,
        seeds_per_dataset=5,
    )
    second = probe.split_seed_sets(
        list(reversed(episode_seeds)),
        holdout_seeds=4,
        validation_holdout_seeds=2,
        seeds_per_dataset=5,
    )

    assert first == second
    assert first.train == (10,)
    assert first.validation == (11, 12)
    assert first.test == (13, 14)
    assert set(first.train).isdisjoint(first.validation)
    assert set(first.train).isdisjoint(first.test)
    assert set(first.validation).isdisjoint(first.test)
    partitions = [set(first.train), set(first.validation), set(first.test)]
    assert set.union(*partitions) == set(episode_seeds)
    assert all(sum(seed in partition for partition in partitions) == 1 for seed in episode_seeds)


def test_prefixes_are_causal_deterministic_and_shuffled_within_peer_groups(probe) -> None:
    # Two corpus/step groups of four rows, with distinct preceding pairs.
    rows, slots = 8, 5
    kinds = np.asarray(
        [
            [(row + 3 * slot) % probe.N_MARKET_KINDS for slot in range(slots)]
            for row in range(rows)
        ],
        dtype=np.int64,
    )
    quantities = np.asarray(
        [
            [(2 * row + slot) % probe.N_QUANTITIES for slot in range(slots)]
            for row in range(rows)
        ],
        dtype=np.int64,
    )
    corpora = np.asarray([0] * 4 + [1] * 4)
    steps = np.asarray([7] * 4 + [2] * 4)
    active = np.asarray(
        [[True, True, row % 2 == 1, False, False] for row in range(rows)],
        dtype=np.bool_,
    )
    episode_seeds = np.asarray([1, 1, 2, 2, 3, 3, 4, 4])

    true = probe.construct_prefixes(
        kinds,
        quantities,
        corpora,
        steps,
        active,
        episode_seeds,
        shuffled=False,
        seed=123,
    )
    shuffled = probe.construct_prefixes(
        kinds,
        quantities,
        corpora,
        steps,
        active,
        episode_seeds,
        shuffled=True,
        seed=123,
    )
    repeated = probe.construct_prefixes(
        kinds,
        quantities,
        corpora,
        steps,
        active,
        episode_seeds,
        shuffled=True,
        seed=123,
    )

    for slot in range(slots):
        assert np.all(true.kinds[:, slot, 0] == probe.N_MARKET_KINDS)
        assert np.all(true.quantities[:, slot, 0] == probe.N_QUANTITIES)
        assert np.array_equal(
            true.kinds[:, slot, 1 : slot + 1], kinds[:, :slot]
        )
        assert np.array_equal(
            true.quantities[:, slot, 1 : slot + 1], quantities[:, :slot]
        )
    assert np.array_equal(shuffled.kinds, repeated.kinds)
    assert np.array_equal(shuffled.quantities, repeated.quantities)
    assert np.array_equal(shuffled.source_rows, repeated.source_rows)
    assert np.all(shuffled.source_rows[:, 1:] != np.arange(rows)[:, None])
    assert not np.array_equal(shuffled.kinds, true.kinds)

    for row in range(rows):
        for slot in range(1, slots):
            source = shuffled.source_rows[row, slot]
            assert corpora[source] == corpora[row]
            assert steps[source] == steps[row]
            assert episode_seeds[source] != episode_seeds[row]
            assert np.array_equal(
                active[source, : slot + 1], active[row, : slot + 1]
            )
            assert np.array_equal(
                shuffled.kinds[row, slot, 1 : slot + 1],
                kinds[source, :slot],
            )
            assert np.array_equal(
                shuffled.quantities[row, slot, 1 : slot + 1],
                quantities[source, :slot],
            )

    # Changing a current/future label cannot alter that slot's already-built prefix.
    changed = kinds.copy()
    changed[:, 3:] = (changed[:, 3:] + 1) % probe.N_MARKET_KINDS
    changed_active = active.copy()
    changed_active[:, 4] = ~changed_active[:, 4]
    no_leak = probe.construct_prefixes(
        changed,
        quantities,
        corpora,
        steps,
        changed_active,
        episode_seeds,
        shuffled=True,
        seed=123,
    )
    assert np.array_equal(no_leak.kinds[:, :4], shuffled.kinds[:, :4])
    assert np.array_equal(
        no_leak.source_rows[:, :4], shuffled.source_rows[:, :4]
    )


def test_paired_episode_aggregation_preserves_true_minus_control_sign(probe) -> None:
    true_nll = np.asarray([[1.0, 3.0], [2.0, 4.0], [0.5, 1.5]])
    control_nll = true_nll + 1.0
    episode_ids = np.asarray([0, 0, 1])
    active = np.ones_like(true_nll, dtype=bool)

    summary = probe.paired_episode_summary(
        true_nll,
        control_nll,
        episode_ids,
        active,
        seed=91,
        bootstrap_samples=200,
    )

    assert summary["episodes"] == 2
    assert summary["nll_true_minus_shuffled"] == pytest.approx(-1.0)
    assert summary["nll_improvement_shuffled_minus_true"] == pytest.approx(1.0)
    assert summary["relative_nll_improvement"] > 0
    assert summary["bootstrap_95_ci_true_minus_shuffled"] == pytest.approx([-1.0, -1.0])


def test_empty_and_invalid_inputs_are_rejected_clearly(probe, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty seed"):
        probe.split_seed_sets([], holdout_seeds=2, validation_holdout_seeds=1)
    with pytest.raises(ValueError, match="smaller than holdout"):
        probe.split_seed_sets(
            [1, 2, 3], holdout_seeds=2, validation_holdout_seeds=2
        )
    with pytest.raises(ValueError, match="empty actions"):
        probe.construct_prefixes(
            np.empty((0, 3), dtype=np.int64),
            np.empty((0, 3), dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=bool),
            np.empty(0, dtype=np.int64),
            shuffled=True,
            seed=0,
        )
    with pytest.raises(ValueError, match="no dataset directories"):
        probe.load_probe_dataset(
            [],
            holdout_seeds=2,
            validation_holdout_seeds=1,
            seeds_per_dataset=None,
            encoded_cache=None,
            encode_workers=1,
            torch_threads=1,
        )
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "manifest.json").write_text(
        '{\"format_version\": 1, \"episode_steps\": 10, \"episodes\": []}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lists no episodes"):
        probe.load_probe_dataset(
            [empty],
            holdout_seeds=2,
            validation_holdout_seeds=1,
            seeds_per_dataset=None,
            encoded_cache=None,
            encode_workers=1,
            torch_threads=1,
        )
