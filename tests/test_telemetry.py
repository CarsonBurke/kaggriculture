from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from kaggriculture import rollout as rollout_module
from kaggriculture import telemetry
from kaggriculture.rollout import RolloutBatch
from kaggriculture.telemetry import (
    TENSORBOARD_MIRROR_FORMAT_VERSION,
    TensorboardMirror,
    migrate_jsonl_to_tensorboard,
    training_step_field,
)
from kaggriculture.training import rollout_diagnostics


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _tags(log_dir: Path) -> list[str]:
    return EventAccumulator(str(log_dir)).Reload().Tags()["scalars"]


def _scalars(log_dir: Path, tag: str) -> list[tuple[int, float]]:
    accumulator = EventAccumulator(str(log_dir)).Reload()
    return [(event.step, event.value) for event in accumulator.Scalars(tag)]


def _training_script():
    path = Path(__file__).parents[1] / "scripts" / "train_ppo.py"
    spec = importlib.util.spec_from_file_location("kaggriculture_train_ppo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The population every population check is exercised at: the plan's default, and
# the number that decides whether the per-agent facets fit. Every category is
# reported once per agent, so a facet joining the chart name instead of the
# category would multiply each accordion by this.
_POPULATION = 4

# The largest population the matrix layout claims to hold for. A row of the
# head-to-head matrix is N-1 charts, so ten is the last size whose rows fit the
# nine-chart budget; the layout says so in a comment, and the budget check below
# is where that claim is actually measured.
_LARGEST_BUDGETED_POPULATION = 10


def test_training_jsonl_migration_is_idempotent_and_rebuilds_after_append(tmp_path: Path) -> None:
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    records = [
        {"iteration": 1, "value_loss": 0.5, "label": "ignored"},
        {"iteration": 2, "value_loss": 0.25},
    ]
    _write_jsonl(journal, records)

    first = migrate_jsonl_to_tensorboard(journal, log_dir)
    second = migrate_jsonl_to_tensorboard(journal, log_dir)

    assert first.rebuilt
    assert not second.rebuilt
    assert _scalars(log_dir, "critic/value_loss") == [(1, 0.5), (2, 0.25)]
    manifest = json.loads((log_dir / ".kaggriculture-tensorboard.json").read_text())
    assert manifest["source"]["path"] == str(journal.resolve())

    records.append({"iteration": 3, "value_loss": 0.125})
    _write_jsonl(journal, records)
    updated = migrate_jsonl_to_tensorboard(journal, log_dir)

    assert updated.rebuilt
    assert _scalars(log_dir, "critic/value_loss") == [(1, 0.5), (2, 0.25), (3, 0.125)]


def test_migration_repairs_a_corrupted_event_file_from_jsonl(tmp_path: Path) -> None:
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    _write_jsonl(journal, [{"iteration": 1, "value_loss": 0.75}])
    migrate_jsonl_to_tensorboard(journal, log_dir)
    event_file = next(log_dir.glob("events.out.tfevents.*"))
    with event_file.open("ab") as stream:
        stream.write(b"corruption")

    repaired = migrate_jsonl_to_tensorboard(journal, log_dir)

    assert repaired.rebuilt
    assert _scalars(log_dir, "critic/value_loss") == [(1, 0.75)]


def test_migration_rejects_a_destination_containing_its_source(tmp_path: Path) -> None:
    log_dir = tmp_path / "tensorboard"
    log_dir.mkdir()
    journal = log_dir / "metrics.jsonl"
    _write_jsonl(journal, [{"iteration": 1, "value_loss": 0.75}])

    with pytest.raises(ValueError, match="cannot contain"):
        migrate_jsonl_to_tensorboard(journal, log_dir)


def test_migration_rejects_a_symlink_destination(tmp_path: Path) -> None:
    journal = tmp_path / "metrics.jsonl"
    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "tensorboard"
    symlink.symlink_to(target, target_is_directory=True)
    _write_jsonl(journal, [{"iteration": 1, "value_loss": 0.75}])

    with pytest.raises(ValueError, match="cannot be a symlink"):
        migrate_jsonl_to_tensorboard(journal, symlink)


def test_migration_rejects_missing_source_and_unowned_destination(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        migrate_jsonl_to_tensorboard(tmp_path / "missing.jsonl", tmp_path / "tensorboard")

    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "important"
    log_dir.mkdir()
    important = log_dir / "do-not-delete.txt"
    important.write_text("valuable", encoding="utf-8")
    _write_jsonl(journal, [{"iteration": 1, "value_loss": 0.75}])

    with pytest.raises(ValueError, match="not owned"):
        migrate_jsonl_to_tensorboard(journal, log_dir)
    assert important.read_text(encoding="utf-8") == "valuable"


def test_migration_recovers_only_a_torn_final_record(tmp_path: Path) -> None:
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    journal.write_bytes(b'{"iteration":1,"value_loss":0.75}\n{"iteration":2')

    migrated = migrate_jsonl_to_tensorboard(journal, log_dir)

    assert migrated.records == 1
    assert _scalars(log_dir, "critic/value_loss") == [(1, 0.75)]

    journal.write_bytes(b'{"iteration":1}\n{"broken"\n{"iteration":3}\n')
    with pytest.raises(ValueError, match="invalid metrics record"):
        migrate_jsonl_to_tensorboard(journal, log_dir)


def test_failed_rebuild_preserves_the_current_mirror(tmp_path: Path) -> None:
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    _write_jsonl(journal, [{"iteration": 1, "value_loss": 0.75}])
    migrate_jsonl_to_tensorboard(journal, log_dir)

    def fail_to_create_writer(path: Path):
        raise OSError(f"simulated failure in {path}")

    with pytest.raises(OSError, match="simulated failure"):
        migrate_jsonl_to_tensorboard(
            journal,
            log_dir,
            force=True,
            writer_factory=fail_to_create_writer,
        )

    assert _scalars(log_dir, "critic/value_loss") == [(1, 0.75)]


def test_live_mirror_recovers_a_committed_record(tmp_path: Path) -> None:
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    first = {"iteration": 1, "value_loss": 0.5}
    second = {"iteration": 2, "value_loss": 0.25}
    _write_jsonl(journal, [first])
    mirror = TensorboardMirror(journal, log_dir)
    _write_jsonl(journal, [first, second])

    mirror.record(second)
    mirror.close()

    assert _scalars(log_dir, "critic/value_loss") == [(1, 0.5), (2, 0.25)]
    assert not migrate_jsonl_to_tensorboard(journal, log_dir).rebuilt


def test_live_mirror_survives_an_equal_length_torn_tail_replacement(tmp_path: Path) -> None:
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    first = {"iteration": 1, "value_loss": 0.5}
    second = {"iteration": 2, "value_loss": 0.25}
    _write_jsonl(journal, [first])
    with journal.open("ab") as stream:
        stream.write(json.dumps(second, sort_keys=True).encode("utf-8") + b"X")
    mirror = TensorboardMirror(journal, log_dir)
    torn_size = journal.stat().st_size

    # The journal appender truncates the torn suffix and commits a complete
    # record of exactly the same byte length, so file size alone cannot reveal
    # the change to the incremental mirror state.
    _write_jsonl(journal, [first, second])
    assert journal.stat().st_size == torn_size

    mirror.record(second)
    mirror.close()

    assert _scalars(log_dir, "critic/value_loss") == [(1, 0.5), (2, 0.25)]


def test_live_mirror_repairs_corruption_before_recording(tmp_path: Path) -> None:
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    first = {"iteration": 1, "value_loss": 0.5}
    second = {"iteration": 2, "value_loss": 0.25}
    _write_jsonl(journal, [first])
    mirror = TensorboardMirror(journal, log_dir)
    event_file = next(log_dir.glob("events.out.tfevents.*"))
    with event_file.open("ab") as stream:
        stream.write(b"corruption")
    _write_jsonl(journal, [first, second])

    mirror.record(second)
    mirror.close()

    assert _scalars(log_dir, "critic/value_loss") == [(1, 0.5), (2, 0.25)]


def test_live_mirror_self_heals_after_a_writer_failure(tmp_path: Path) -> None:
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    first = {"iteration": 1, "value_loss": 0.5}
    second = {"iteration": 2, "value_loss": 0.25}
    _write_jsonl(journal, [first])
    failed = False

    class FailingWriter:
        def __init__(self, writer: SummaryWriter) -> None:
            self.writer = writer

        def add_scalar(self, *args, **kwargs) -> None:
            raise OSError("simulated event-file failure")

        def add_text(self, *args, **kwargs) -> None:
            self.writer.add_text(*args, **kwargs)

        def flush(self) -> None:
            self.writer.flush()

        def close(self) -> None:
            self.writer.close()

    def factory(path: Path):
        nonlocal failed
        writer = SummaryWriter(path)
        if path.resolve() == log_dir.resolve() and not failed:
            failed = True
            return FailingWriter(writer)
        return writer

    mirror = TensorboardMirror(journal, log_dir, writer_factory=factory)
    _write_jsonl(journal, [first, second])

    mirror.record(second)
    mirror.close()

    assert failed
    assert _scalars(log_dir, "critic/value_loss") == [(1, 0.5), (2, 0.25)]


def test_benchmark_batches_are_runs_sharing_the_training_categories(tmp_path: Path) -> None:
    """A benchmark iteration is a training iteration, so it charts like one.

    Its scalars used to be flattened into `{kind}/{mode}/games_{n}/{name}`
    tags, which put all 63 of them under one first path component -- the whole
    benchmark mirror opened as a single accordion. Making the batch a run
    instead gives every metric the category it already has in a training
    journal, and puts the batch sizes on one chart as separate series, which is
    the comparison a sweep is run for.
    """
    journal = tmp_path / "rollout.jsonl"
    log_dir = tmp_path / "tensorboard"
    _write_jsonl(
        journal,
        [
            {"event": "configuration", "compile_models": True, "games": [16]},
            {
                "event": "repeat",
                "games": 16,
                "repeat": 0,
                "total_seconds": 4.5,
                "update_replay_unit_kl": 0.25,
                "some_metric_added_later": 0.125,
            },
            {
                "event": "batch_summary",
                "games": 16,
                "steady_total_seconds_median": 5.0,
            },
        ],
    )

    migrate_jsonl_to_tensorboard(journal, log_dir)

    assert _scalars(log_dir / "rollout-compiled/games_16", "timing/total_seconds") == [(0, 4.5)]
    # A per-head run composes under the batch rather than escaping to the root,
    # so two batches cannot write one head's series over each other.
    assert _scalars(log_dir / "rollout-compiled/games_16", "parity-unit/kl") == [(0, 0.25)]
    assert _scalars(log_dir / "rollout-compiled/games_16", "misc/some_metric_added_later") == [
        (0, 0.125)
    ]
    # The summary is a curve over batch size, so it stays one run stepped by
    # the game count rather than splitting per batch like the repeats do.
    assert _scalars(log_dir / "rollout-compiled/batch_summary", "steady/total_seconds_median") == [
        (16, 5.0)
    ]


def test_epoch_keyed_training_journal_mirrors_as_scalars(tmp_path: Path) -> None:
    """Behavior cloning counts epochs where PPO counts iterations.

    A training record that is not recognized as one falls through to the
    metadata branch and is mirrored as a JSON text blob, so the run's curves
    silently do not appear in TensorBoard at all -- which is exactly how this
    was found.
    """
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    # Exact binary fractions: the event file stores float32, so a decimal
    # literal would come back rounded and the comparison would be about
    # floating point rather than about the mirror.
    records = [
        {"epoch": 0, "train_loss": 0.5, "holdout_nll": 0.75, "holdout_unit_accuracy": 0.875},
        {"epoch": 1, "train_loss": 0.25, "holdout_nll": 0.375, "holdout_unit_accuracy": 0.9375},
    ]
    _write_jsonl(journal, records)

    result = migrate_jsonl_to_tensorboard(journal, log_dir)

    assert result.records == 2
    assert _scalars(log_dir, "loss/train") == [(0, 0.5), (1, 0.25)]
    assert _scalars(log_dir, "loss/holdout_nll") == [(0, 0.75), (1, 0.375)]
    # Per-head holdout statistics are a run each, so unit, kind and quantity
    # share one accuracy chart instead of occupying three.
    assert _scalars(log_dir, "holdout-unit/accuracy") == [(0, 0.875), (1, 0.9375)]
    # The step field itself is a coordinate, not a curve.
    accumulator = EventAccumulator(str(log_dir)).Reload()
    assert "epoch" not in accumulator.Tags()["scalars"]


def test_training_step_field_distinguishes_journals_from_benchmark_records() -> None:
    assert training_step_field({"iteration": 3, "loss": 0.1}) == "iteration"
    assert training_step_field({"epoch": 0, "train_loss": 0.1}) == "epoch"
    # A benchmark record is tagged, and its `repeat`/`games` fields are not a
    # training step even though they are integers.
    assert training_step_field({"event": "iteration", "repeat": 1, "games": 64}) is None
    assert training_step_field({"event": "configuration", "seed": 7}) is None
    assert training_step_field({"loss": 0.1}) is None
    # Booleans are ints in Python; a flag is not a step.
    assert training_step_field({"epoch": True, "loss": 0.1}) is None


def _league_record(iteration: int, opponents: dict[int, dict]) -> dict:
    record: dict = {"iteration": iteration, "league_score_rate": 0.5}
    for identifier, fields in opponents.items():
        for field, value in fields.items():
            record[f"league_opponent_{identifier:08d}_{field}"] = value
    return record


def test_league_opponents_aggregate_instead_of_minting_a_tag_each(tmp_path: Path) -> None:
    """Per-opponent fields are keyed by checkpoint index, which never repeats.

    Mirrored field by field they mint new tags for every opponent the league
    ever samples -- 494 of them by the point this was reported -- and each one
    is a curve defined only on the iterations that single opponent happened to
    be drawn, which is not something a chart can be read against. The aggregate
    over a category is defined on every iteration.
    """
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    _write_jsonl(
        journal,
        [
            _league_record(
                1,
                {
                    4: {"category": "active", "games": 24, "score_rate": 0.5, "mean_margin": 100.0},
                    9: {"category": "active", "games": 8, "score_rate": 0.25, "mean_margin": -50.0},
                    2: {
                        "category": "historical",
                        "games": 16,
                        "score_rate": 0.75,
                        "mean_margin": 8.0,
                    },
                },
            ),
            # A later iteration draws an opponent never seen before. Under the
            # per-opponent layout this is where the namespace grows; under
            # aggregation it is the same four tags with new values.
            _league_record(
                2,
                {
                    31: {"category": "active", "games": 32, "score_rate": 1.0, "mean_margin": 4.0},
                },
            ),
        ],
    )

    migrate_jsonl_to_tensorboard(journal, log_dir)

    active = EventAccumulator(str(log_dir)).Reload().Tags()["scalars"]
    assert {tag for tag in active if tag.startswith("opponents-active/")} == {
        "opponents-active/count",
        "opponents-active/games",
        "opponents-active/score_rate",
        "opponents-active/mean_margin",
        "opponents-active/score_rate_min",
        "opponents-active/score_rate_max",
    }
    # Rates are weighted by games played, so the opponent drawn for 24 games
    # counts for three times the one drawn for 8: (0.5*24 + 0.25*8)/32.
    assert _scalars(log_dir, "opponents-active/score_rate")[0] == (1, 0.4375)
    assert _scalars(log_dir, "opponents-active/mean_margin")[0] == (1, 62.5)
    assert _scalars(log_dir, "opponents-active/count") == [(1, 2.0), (2, 1.0)]
    assert _scalars(log_dir, "opponents-active/score_rate_min")[0] == (1, 0.25)
    assert _scalars(log_dir, "opponents-historical/games") == [(1, 16.0)]

    # No tag anywhere names an individual opponent, whatever its index.
    for path in log_dir.rglob("events.out.tfevents.*"):
        for tag in EventAccumulator(str(path.parent)).Reload().Tags()["scalars"]:
            assert "opponent_" not in tag
            assert not any(character.isdigit() for character in tag)


def test_an_opponent_key_containing_an_underscore_still_resolves(tmp_path: Path) -> None:
    """Built-in keys are names, not zero-padded indices, so they contain `_`.

    Splitting the field off at the FIRST underscore read `builtin_starter` as an
    opponent called `builtin` carrying a field called `starter_score_rate`. That
    collapsed all three built-ins into a single record, classified it as
    unclassified because no field named `category` survived, and dropped the
    score-rate and margin curves entirely -- which are the only curves that say
    whether the league is beating the reference agents it was admitted to beat.
    """
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    record: dict = {"iteration": 1, "league_score_rate": 0.5}
    for key, fields in (
        ("00000009", {"category": "active", "games": 14, "score_rate": 0.5, "mean_margin": 12.0}),
        (
            "builtin_starter",
            {"category": "builtin", "games": 14, "score_rate": 0.0, "mean_margin": -3388.0},
        ),
        (
            "builtin_pass",
            {"category": "builtin", "games": 13, "score_rate": 0.25, "mean_margin": -2900.0},
        ),
    ):
        for field, value in fields.items():
            record[f"league_opponent_{key}_{field}"] = value
    _write_jsonl(journal, [record])

    migrate_jsonl_to_tensorboard(journal, log_dir)

    assert _scalars(log_dir, "opponents-builtin/count") == [(1, 2.0)]
    assert _scalars(log_dir, "opponents-builtin/games") == [(1, 27.0)]
    # Both built-ins are present and weighted by games: (0.0*14 + 0.25*13)/27.
    assert _scalars(log_dir, "opponents-builtin/score_rate")[0][1] == pytest.approx(0.25 * 13 / 27)
    assert _scalars(log_dir, "opponents-builtin/score_rate_min") == [(1, 0.0)]
    assert _scalars(log_dir, "opponents-builtin/score_rate_max") == [(1, 0.25)]
    # The snapshot key beside them, which has no underscore, is unaffected.
    assert _scalars(log_dir, "opponents-active/games") == [(1, 14.0)]
    assert not any(tag.startswith("opponents-unclassified/") for tag in _tags(log_dir))


def test_a_cohort_is_a_category_so_one_mirror_stays_one_event_file(tmp_path: Path) -> None:
    """The wave, its self-play half and its league half report the same things.

    As tag prefixes those are three charts in three categories that have to be
    read side by side; as runs they are three series on one chart, which is the
    comparison the split exists to support.
    """
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    _write_jsonl(
        journal,
        [
            {
                "iteration": 1,
                "score_rate": 0.5,
                "self_play_score_rate": 0.25,
                "league_score_rate": 0.75,
                "unit_move_fraction": 0.125,
                "unit_teleport_fraction": 0.0625,
                "league_money_median": 0.0,
            }
        ],
    )

    migrate_jsonl_to_tensorboard(journal, log_dir)

    # One event file for the whole mirror: the cohort is the category's suffix,
    # not a run of its own, so `outcome-wave` and `outcome-league` sort adjacently
    # in one selector row instead of opening three.
    assert [path.parent for path in log_dir.rglob("events.out.tfevents.*")] == [log_dir]
    for cohort, expected in (("wave", 0.5), ("self-play", 0.25), ("league", 0.75)):
        assert _scalars(log_dir, f"outcome-{cohort}/score_rate") == [(1, expected)]
    # A shipped unit action is filed by what it acts on, and an action the game
    # grows still lands on a chart of its own under the family fallback.
    assert _scalars(log_dir, "logistics-wave/move_fraction") == [(1, 0.125)]
    assert _scalars(log_dir, "unit-actions-wave/teleport_fraction") == [(1, 0.0625)]
    # The family token is absorbed by the category where it would only stutter,
    # and kept where the category holds more than one family.
    assert _scalars(log_dir, "economy-league/money_median") == [(1, 0.0)]


def _pair_value(agent: int, opponent: int) -> float:
    """A distinct reading per pair, exact in binary so the event file returns it."""
    return agent / 8.0 + opponent / 64.0


def _population_record(iteration: int) -> dict:
    """One iteration of a population run, as the training loop journals it.

    The disagreement matrix is written for every ordered pair, which is how a
    caller holding a symmetric matrix would walk it; the builder folds each pair
    onto one field, so the record carries six entries rather than twelve.
    """
    record: dict = {"iteration": iteration, "score_rate": 0.5}
    for index in range(_POPULATION):
        record[telemetry.population_agent_field(index, "value_loss")] = 0.5 * index
        record[telemetry.population_agent_field(index, "score_rate")] = 0.25 * index
    for agent in range(_POPULATION):
        for opponent in range(_POPULATION):
            if agent == opponent:
                continue
            record[telemetry.population_head_to_head_field(agent, opponent)] = _pair_value(
                agent, opponent
            )
            record[telemetry.population_disagreement_pair_field(agent, opponent)] = _pair_value(
                min(agent, opponent), max(agent, opponent)
            )
    record[telemetry.population_disagreement_field("mean")] = 0.75
    record[telemetry.population_disagreement_field("min")] = 0.5
    return record


def test_a_symmetric_matrix_is_named_once_and_neither_matrix_has_a_diagonal() -> None:
    """The two builders differ exactly where the two matrices differ.

    Disagreement is symmetric, so both orderings of a pair name one field and the
    reading is published once; a score rate is asymmetric, so they name two. And
    neither matrix has a diagonal to publish -- `rollout.population_pairings`
    seats no member against itself and a policy's disagreement with itself is zero
    by construction -- so a `vs_agent2` chart in agent 2's own row could only be a
    caller's bug, arriving as a measurement.
    """
    assert telemetry.population_disagreement_pair_field(
        3, 1
    ) == telemetry.population_disagreement_pair_field(1, 3)
    assert telemetry.population_head_to_head_field(3, 1) != telemetry.population_head_to_head_field(
        1, 3
    )
    for builder in (
        telemetry.population_head_to_head_field,
        telemetry.population_disagreement_pair_field,
    ):
        with pytest.raises(ValueError):
            builder(2, 2)


def test_a_population_reading_nobody_filed_stays_visible() -> None:
    """A reading added later must land where a person will notice it.

    The matrices are recognized by name and only their indices are parsed, so a
    two-word reading keeps its whole name as the chart and a field naming neither
    matrix keeps the bare category. Recovering the matrix from the end of the name
    instead would file `disagreement_mean_first_iteration` under
    `population-disagreement-mean`, one sort position from the real category and
    indistinguishable from a category somebody meant.
    """
    reading = telemetry.population_disagreement_field("mean_first_iteration")
    assert telemetry._placement(reading) == ("", "population-disagreement/mean_first_iteration")
    assert telemetry._placement("population_cycle_length") == ("", "population/cycle_length")


def test_nextlat_actor_and_critic_metrics_have_deliberate_separate_categories() -> None:
    expected = {
        "structured_preupdate_decision": "nextlat-actor-holdout-decision/decision",
        "structured_predictor_combined": "nextlat-actor-predictor/combined",
        "structured_persistence_combined_ratio": "nextlat-actor-persistence/combined_ratio",
        "structured_persistence_decision_informative": (
            "nextlat-actor-persistence/decision_informative"
        ),
        "structured_actor_decision": "nextlat-actor-auxiliary-decision/decision",
        "structured_critic_preupdate_value": "nextlat-critic-holdout-loss/value",
        "structured_critic_predictor_latent": "nextlat-critic-predictor-loss/latent",
        "structured_critic_persistence_combined_ratio": (
            "nextlat-critic-persistence/combined_ratio"
        ),
        "structured_critic_persistence_value_informative": (
            "nextlat-critic-persistence/value_informative"
        ),
        "structured_critic_value": "nextlat-critic-auxiliary-loss/value",
        "structured_learning_rate": "nextlat-schedule/actor_learning_rate",
        "structured_critic_learning_rate": "nextlat-schedule/critic_learning_rate",
        "structured_gradient_source_norm": "nextlat-actor-gradients/source_norm",
        "structured_critic_gradient_source_norm": ("nextlat-critic-gradients/source_norm"),
        "credit_preupdate_opponent_scripted_v27_ttg_33_128_terminal_residual_mse": (
            "credit-opponents-ttg-33-128/terminal_residual_mse"
        ),
    }
    for field, tag in expected.items():
        assert telemetry._placement(field) == ("", tag)
        assert telemetry._placement(f"agent3_{field}") == (
            "",
            f"{tag.partition('/')[0]}-agent3/{tag.partition('/')[2]}",
        )


def test_opponent_credit_metrics_are_aggregated_into_horizon_cohorts(tmp_path: Path) -> None:
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    metrics = {}
    for group, states, mse in (
        ("opponent_00000095", 2, 1.0),
        ("opponent_builtin_starter", 6, 3.0),
    ):
        prefix = f"credit_preupdate_{group}_all"
        metrics.update(
            {
                f"{prefix}_states": states,
                f"{prefix}_mc_mse": mse,
                f"{prefix}_potential_only_mse": mse + 1.0,
                f"{prefix}_terminal_residual_mse": mse + 2.0,
                f"{prefix}_terminal_target_variance": mse + 3.0,
                f"{prefix}_terminal_residual_explained_variance": mse / 10.0,
            }
        )
    _write_jsonl(journal, [{"iteration": 1, **metrics}])

    migrate_jsonl_to_tensorboard(journal, log_dir)

    tags = set(_tags(log_dir))
    assert "credit-opponents-all/states" in tags
    assert "credit-opponents-all/mc_mse" in tags
    assert not any("credit-opponent-" in tag for tag in tags)
    assert _scalars(log_dir, "credit-opponents-all/states") == [(1, 8.0)]
    assert _scalars(log_dir, "credit-opponents-all/mc_mse") == [(1, 2.5)]
    assert _scalars(log_dir, "credit-opponents-all/terminal_residual_mse") == [(1, 4.5)]


def test_a_population_run_facets_every_agent_and_stays_one_event_file(tmp_path: Path) -> None:
    """Four concurrent learners report the same categories four times over.

    The agent joins the category, so `critic-agent0` is an accordion of the size
    `critic` has for a single learner, and the wave's own reading keeps the tag it
    always had. A facet dropped in any one branch of the placement rules would put
    one agent's curve on top of that shared tag, where two series at one step
    render as a single plausible noisy line.
    """
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    records = [_population_record(iteration) for iteration in (1, 2)]

    # Through the live mirror rather than a rebuild, because that is the writer a
    # run actually uses and the one that would open a directory per agent. The
    # journal is appended to before each record, as the loop appends before it
    # mirrors.
    mirror = TensorboardMirror(journal, log_dir)
    for index, record in enumerate(records, start=1):
        _write_jsonl(journal, records[:index])
        mirror.record(record)
    mirror.close()

    # Four learners, still one event file: no run is opened per agent, so a parent
    # `--logdir` shows one selector row for the run however large N is.
    assert [path.parent for path in log_dir.rglob("events.out.tfevents.*")] == [log_dir]
    steps = (1, 2)
    assert _scalars(log_dir, "outcome-wave/score_rate") == [(step, 0.5) for step in steps]
    for index in range(_POPULATION):
        assert _scalars(log_dir, f"critic-agent{index}/value_loss") == [
            (step, 0.5 * index) for step in steps
        ]
        assert _scalars(log_dir, f"outcome-wave-agent{index}/score_rate") == [
            (step, 0.25 * index) for step in steps
        ]
    # Nothing per-agent fell through to the category of last resort.
    assert not any(tag.startswith("misc") for tag in _tags(log_dir))


def test_the_population_category_carries_the_matrix_and_both_disagreement_readings(
    tmp_path: Path,
) -> None:
    """Cycling and convergence are visible here and nowhere else.

    Every reward in a population wave is relative, so a three-cycle among four
    learners and a population that has collapsed into one policy both read 0.5 on
    every other chart. The score rate is asymmetric -- seat and opponent order
    both move it -- so all twelve ordered pairs are charted; the disagreement
    matrix is symmetric with a zero diagonal, so its six distinct pairs are the
    whole of it and the mirror image is not written at all.
    """
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    _write_jsonl(journal, [_population_record(1)])

    migrate_jsonl_to_tensorboard(journal, log_dir)

    for agent in range(_POPULATION):
        for opponent in range(_POPULATION):
            if agent == opponent:
                continue
            assert _scalars(log_dir, f"population-score-rate-agent{agent}/vs_agent{opponent}") == [
                (1, _pair_value(agent, opponent))
            ]
    assert {tag for tag in _tags(log_dir) if tag.startswith("population-disagreement-")} == {
        f"population-disagreement-agent{low}/vs_agent{high}"
        for low in range(_POPULATION)
        for high in range(low + 1, _POPULATION)
    }
    for low in range(_POPULATION):
        for high in range(low + 1, _POPULATION):
            assert _scalars(log_dir, f"population-disagreement-agent{low}/vs_agent{high}") == [
                (1, _pair_value(low, high))
            ]
    # The two readings the convergence gate is set from, in the matrix's own
    # category rather than one agent's facet: neither belongs to a member.
    assert _scalars(log_dir, "population-disagreement/mean") == [(1, 0.75)]
    assert _scalars(log_dir, "population-disagreement/min") == [(1, 0.5)]


def test_every_field_is_placed_and_an_unrecognized_one_stays_visible(tmp_path: Path) -> None:
    """A field nobody filed must land somewhere a person will notice it.

    Dropping it would be worse than filing it badly: the curve would simply be
    absent from TensorBoard with nothing to indicate it had ever been written,
    which is indistinguishable from the metric not being computed at all.
    """
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    _write_jsonl(
        journal,
        [{"iteration": 1, "some_metric_added_later": 0.5, "next_seed": 12345, "value_loss": 0.25}],
    )

    migrate_jsonl_to_tensorboard(journal, log_dir)

    tags = EventAccumulator(str(log_dir)).Reload().Tags()["scalars"]
    assert "misc/some_metric_added_later" in tags
    # A resume token is a coordinate for the next run, not a curve about this
    # one, and it is the one field deliberately not plotted.
    assert not any("next_seed" in tag for tag in tags)


def test_flat_bookkeeping_and_thresholds_are_not_plotted(tmp_path: Path) -> None:
    """Constants cost charts and say nothing, so the mirror leaves them out.

    Schedule hyperparameters are flat within a run, per-head abort thresholds
    are flat configuration bounds, and `throughput/states` repeats the wave
    cohort's `rollout_states` exactly. The journal keeps every one of them --
    only the TensorBoard rows go away -- so this asserts the tags are absent
    while their journal values survive a migration round-trip.
    """
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    _write_jsonl(
        journal,
        [
            {
                "iteration": index,
                "value_loss": 0.25,
                "epochs": 2,
                "gamma": 0.997,
                "actor_gae_lambda": 0.97,
                "critic_gae_lambda": 1.0,
                "states": 1000,
                "rollout_states": 1000,
                "update_replay_unit_kl_fatal_at": 0.005,
                "update_replay_unit_tail_fraction_fatal_at": 0.0002,
            }
            for index in range(3)
        ],
    )

    migrate_jsonl_to_tensorboard(journal, log_dir)

    tags = _tags(log_dir)
    assert "critic/value_loss" in tags
    assert "rollout-wave/states" in tags
    for absent in (
        "schedule/epochs",
        "schedule/gamma",
        "schedule/actor_gae_lambda",
        "schedule/critic_gae_lambda",
        "throughput/states",
        # `states` must not survive as an unfiled leftover either.
        "misc/states",
        "parity-unit/kl_fatal_at",
        "parity-unit/tail_fraction_fatal_at",
    ):
        assert absent not in tags


def test_representation_diagnostics_file_under_their_own_module(tmp_path: Path) -> None:
    """Behavior cloning's belief statistics are curves, not unfiled leftovers.

    They match no NextLat prefix, so they used to land in `misc` -- twenty-eight
    charts in the drawer for metrics nobody filed. Each belief module is now one
    accordion of its four statistics, and a module the game grows later still
    lands in `misc`, visible, until it is filed deliberately.
    """
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    _write_jsonl(
        journal,
        [
            {
                "epoch": 0,
                "train_loss": 0.5,
                "structured_own_patches_variance": 0.1,
                "structured_own_patches_effective_rank": 3.0,
                "structured_own_patches_cosine": 0.2,
                "structured_own_patches_dispersion": 0.3,
                "structured_future_module_cosine": 0.4,
            }
        ],
    )

    migrate_jsonl_to_tensorboard(journal, log_dir)

    tags = _tags(log_dir)
    assert "loss/train" in tags
    for statistic in ("variance", "effective_rank", "cosine", "dispersion"):
        assert f"representation-own-patches/{statistic}" in tags
    assert "misc/structured_future_module_cosine" in tags
    assert not any(
        tag.startswith("misc/structured_") and "future_module" not in tag for tag in tags
    )


def test_a_mirror_written_under_an_older_layout_is_rebuilt_not_extended(tmp_path: Path) -> None:
    """Two tag schemes in one directory make every chart unreadable.

    The layout is versioned so an existing mirror is stale rather than
    appendable, because the alternative is a run whose early iterations are
    plotted under one set of tags and whose later ones are plotted under
    another, with no indication on any chart that the break happened.
    """
    journal = tmp_path / "metrics.jsonl"
    log_dir = tmp_path / "tensorboard"
    _write_jsonl(journal, [{"iteration": 1, "value_loss": 0.5}])
    migrate_jsonl_to_tensorboard(journal, log_dir)
    assert not migrate_jsonl_to_tensorboard(journal, log_dir).rebuilt

    manifest_path = log_dir / ".kaggriculture-tensorboard.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format_version"] == TENSORBOARD_MIRROR_FORMAT_VERSION
    manifest["format_version"] = f"{TENSORBOARD_MIRROR_FORMAT_VERSION}-older"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert migrate_jsonl_to_tensorboard(journal, log_dir).rebuilt


def test_renaming_a_tag_invalidates_mirrors_without_a_manual_version_bump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The layout tables are the layout, so they must be the version.

    A hand-maintained number fails silently in exactly one direction: the tag
    changes, nobody bumps, and every existing mirror keeps serving a scheme that
    no longer matches the code that reads it.
    """
    before = telemetry._layout_fingerprint()

    monkeypatch.setitem(telemetry._TRAINING_TAGS, "value_loss", "critic/renamed")
    assert telemetry._layout_fingerprint() != before

    monkeypatch.setitem(telemetry._TRAINING_TAGS, "value_loss", "critic/value_loss")
    assert telemetry._layout_fingerprint() == before


def _mirrored_field_names() -> list[str]:
    """Field names the mirror is handed, taken from the code that writes them.

    Walking `_TRAINING_TAGS` alone is what let the layout checks below pass
    while `unit-actions` held thirteen charts: nothing placed by a prefix rule
    appears in that table. So the behavioral statistics are read off a real
    diagnostics call, and the parity and holdout families are built from the
    same constants the writers build them from, which is what makes the game
    growing an action or the audit growing a statistic show up here.

    It is not a closed set. `update_ppo`'s own metrics are checked against
    their placements where they are produced, in the PPO tests, because running
    an update is the only honest way to enumerate them.
    """
    trajectories, horizon = 2, 3
    statistics = rollout_diagnostics(
        RolloutBatch(
            architecture="conv",
            states={},
            **{
                name: (np.ones if dtype is np.bool_ else np.zeros)(
                    (trajectories, horizon, *shape), dtype=dtype
                )
                for name, (shape, dtype) in rollout_module._SHARED_FIELD_SPECS.items()
            },
            episode_seeds=np.zeros(trajectories, dtype=np.int64),
            final_money=np.zeros(trajectories, dtype=np.float32),
            opponent_money=np.zeros(trajectories, dtype=np.float32),
            seats=np.asarray([0, 1], dtype=np.int8),
            agents=np.zeros(trajectories, dtype=np.int64),
            learner_stochastic=True,
            orientations=np.zeros(trajectories, dtype=np.int8),
            entropy_sums=np.zeros((trajectories, horizon), dtype=np.float32),
            elapsed_seconds=1.0,
        )
    )
    # Everything placed by a prefix rule rather than by the table: the parity
    # audit's per-head statistics, behavior cloning's per-head holdout scores,
    # and the per-head abort thresholds. The thresholds are journal fields like
    # the rest, so they stay in this enumeration; that they are unplotted by
    # design is asserted by the dedicated test below. These are the families
    # the first version of this
    # helper still missed -- 26 parity fields and the whole holdout set -- so
    # adding two entries to `PARITY_STATISTICS` could push every `parity/<head>`
    # run over budget with both checks green.
    training = _training_script()
    parity = [
        f"update_replay_{component}_{statistic}{suffix}"
        for component in training.PARITY_COMPONENTS
        for statistic, _bound, _description in training.PARITY_STATISTICS
        for suffix in ("", "_fatal_at")
    ]
    parity += [
        f"update_replay_{component}_{statistic}"
        for component in training.PARITY_COMPONENTS
        for statistic in ("active_count", "logprob_max_abs_error", "ratio_max_abs_error")
    ]
    holdout = [
        f"holdout_{head}_{statistic}"
        for head in ("unit", "kind", "quantity")
        for statistic in ("accuracy", "nll", "entropy")
    ]
    # A cohort prefix of "" makes the wave's names identical to the root
    # training fields it shares, so the same name arrives twice and is one
    # field rather than two competing for a tag.
    return list(
        dict.fromkeys(
            (
                *telemetry._TRAINING_TAGS,
                *(prefix + name for prefix, _run in telemetry._COHORT_RUNS for name in statistics),
                *parity,
                *holdout,
            )
        )
    )


def _population_field_names(population: int = _POPULATION) -> list[str]:
    """Field names a population run adds to the ones above.

    A per-agent twin of every field, both matrices in full, and the whole-
    population disagreement readings. Every one is built with the helper the
    training loop builds it with, so a change to a prefix moves both sides or
    fails here, rather than filing one agent's curves in `misc` while every other
    check stays green.

    `dict.fromkeys` because the disagreement matrix is symmetric: walking every
    ordered pair asks for each distinct pair twice and is answered with one field
    name, which is the property that keeps the mirror image off the charts.
    """
    shared = _mirrored_field_names()
    return list(
        dict.fromkeys(
            (
                *(
                    telemetry.population_agent_field(index, name)
                    for index in range(population)
                    for name in shared
                ),
                *(
                    builder(agent, opponent)
                    for builder in (
                        telemetry.population_head_to_head_field,
                        telemetry.population_disagreement_pair_field,
                    )
                    for agent in range(population)
                    for opponent in range(population)
                    if agent != opponent
                ),
                *(
                    telemetry.population_disagreement_field(reading)
                    # `floor` is the iteration-0 reference the convergence gate is
                    # set from, charted beside the readings it bounds.
                    for reading in ("mean", "min", "floor")
                ),
            )
        )
    )


def test_no_two_fields_share_one_run_and_tag() -> None:
    """A collision is silent: two series at the same step render as one line.

    Nothing raises, nothing is dropped, and the chart shows a plausible noisy
    curve that is actually two different measurements interleaved.
    """
    seen: dict[tuple[str, str], str] = {}
    for name in (*_mirrored_field_names(), *_population_field_names()):
        placement = telemetry._placement(name)
        if placement is None:
            continue
        collided = seen.get(placement)
        assert collided is None, f"{name} and {collided} both write {placement}"
        seen[placement] = name


def test_no_category_grows_past_the_readable_budget() -> None:
    """A category is a scrollable accordion in TensorBoard, not a chart.

    Past about nine charts it stops being scannable and the reason to have
    categories at all is gone, which is how the league tags reached 494 and how
    every unit action ended up in one accordion. The budget is per run as well
    as per category: `rollout` under a cohort and `rollout` under root are two
    different accordions and each gets the full allowance.

    A benchmark journal is counted under a batch run because it is the other
    thing this mirror writes, and it is where the worst of this was: its
    scalars used to be tagged `{kind}/{mode}/games_{n}/{name}`, which is one
    category holding all 63 of them.

    A population of four is counted alongside the single learner, because that
    is where a misplaced facet is fatal rather than merely untidy: the agent
    joins the category, so each of its accordions keeps the size it has for one
    learner. Joining the chart name instead would collect every agent's copy
    under one first path segment -- `parity-unit` from seven charts to
    thirty-five, and `population-score-rate` over budget on the ordered pairs
    alone, twelve of them at four agents. Ten is counted as well, being the size
    the matrix layout claims to hold to: a row is N-1 charts, so that claim is a
    measurement here and not a hope.
    """
    benchmark_run = telemetry._BENCHMARK_BATCH_RUN.format(kind="ppo", mode="eager", games=112)
    populations = (_POPULATION, _LARGEST_BUDGETED_POPULATION)
    names = dict.fromkeys(
        (
            *_mirrored_field_names(),
            *(name for size in populations for name in _population_field_names(size)),
        )
    )
    counts: dict[tuple[str, str], int] = {}
    for name in names:
        placement = telemetry._placement(name)
        if placement is None:
            continue
        run, tag = placement
        for root in ("", benchmark_run):
            category = (f"{root}/{run}" if root and run else root or run, tag.split("/")[0])
            counts[category] = counts.get(category, 0) + 1

    oversized = {name: count for name, count in counts.items() if count > 9}
    assert not oversized, oversized


def test_each_producer_is_labeled_by_the_knobs_it_actually_records() -> None:
    """Three producers name their compilation differently, so one key cannot
    label all of them.

    The PPO iteration benchmark decides a collection phase and an update phase
    separately, carrying `rollout_forward_mode` at top level and
    `update_compile_mode` nested under `ppo`. The replay-parity audit carries
    both at top level and has no `self_play_game_counts`. The rollout benchmark
    has only the collector and carries `compile_models`. Reading one key for all
    three does not raise -- it silently labels the others "eager", which drops
    unlike runs into one TensorBoard series where the difference reads as noise
    rather than as the knob it is.

    Both halves are modes, so both are named rather than flagged: filing
    `cudagraphs` and `inductor` together would hide a difference larger than
    either one's difference from eager, 5.309 ms against 2.720 ms against
    4.907 ms on the isolated fp32 forward, and filing `default` beside
    `max-autotune` would hide whether a win came from graph capture or from
    benchmarked kernel selection.
    """
    ppo = {
        "event": "configuration",
        "self_play_game_counts": [112],
        "rollout_forward_mode": "inductor",
        "ppo": {"update_compile_mode": "default"},
    }
    audit = {
        "event": "configuration",
        "self_play_games": 112,
        "rollout_forward_mode": "inductor",
        "update_compile_mode": "default",
    }
    rollout = {"event": "configuration", "games": [16], "compile_models": True}

    assert telemetry._configuration_context(ppo) == ("ppo", "rollout-inductor-update-default")
    assert telemetry._configuration_context(audit) == ("audit", "rollout-inductor-update-default")
    assert telemetry._configuration_context(rollout) == ("rollout", "compiled")

    # The mixed pairings are the point of the split, and the eager-collector
    # against compiled-update one is what an update-only decision runs. Labeling
    # it "eager" would file a run whose update phase is ~1.8x faster alongside a
    # genuinely eager one. The compiling modes are equally unmixable, on both
    # sides.
    for record, expected in (
        ({**ppo, "rollout_forward_mode": "eager"}, "rollout-eager-update-default"),
        ({**ppo, "ppo": {"update_compile_mode": "eager"}}, "rollout-inductor-update-eager"),
        (
            {**ppo, "ppo": {"update_compile_mode": "max-autotune"}},
            "rollout-inductor-update-max-autotune",
        ),
        ({**ppo, "rollout_forward_mode": "cudagraphs"}, "rollout-cudagraphs-update-default"),
        ({**audit, "rollout_forward_mode": "eager"}, "rollout-eager-update-default"),
        ({**audit, "update_compile_mode": "eager"}, "rollout-inductor-update-eager"),
        (
            {**audit, "update_compile_mode": "reduce-overhead"},
            "rollout-inductor-update-reduce-overhead",
        ),
        ({**audit, "rollout_forward_mode": "cudagraphs"}, "rollout-cudagraphs-update-default"),
    ):
        assert telemetry._configuration_context(record)[1] == expected

    assert telemetry._configuration_context(
        {**ppo, "rollout_forward_mode": "eager", "ppo": {"update_compile_mode": "eager"}}
    ) == ("ppo", "rollout-eager-update-eager")
    assert telemetry._configuration_context({**rollout, "compile_models": False}) == (
        "rollout",
        "eager",
    )

    # No producer may be labeled by another's key. A report carrying only the
    # wrong one is what a half-finished rename produces, and reading it would
    # report a compiled run as eager.
    assert telemetry._configuration_context(
        {"event": "configuration", "self_play_game_counts": [112], "compile_models": True}
    ) == ("ppo", "rollout-eager-update-eager")
    assert telemetry._configuration_context(
        {"event": "configuration", "games": [16], "rollout_forward_mode": "inductor"}
    ) == ("rollout", "eager")
    # The ppo kind must not read `update_compile_mode` from top level: that is
    # the audit's schema, and the two are distinguished precisely so this does
    # not silently succeed.
    assert telemetry._configuration_context(
        {
            "event": "configuration",
            "self_play_game_counts": [112],
            "update_compile_mode": "max-autotune",
        }
    ) == ("ppo", "rollout-eager-update-eager")
    # A mode that is not a mode name is a label problem, not a mirror failure:
    # the domain is enforced where a report becomes a decision. Either side.
    assert telemetry._configuration_context({**ppo, "rollout_forward_mode": True}) == (
        "ppo",
        "rollout-eager-update-default",
    )
    assert telemetry._configuration_context({**audit, "update_compile_mode": True}) == (
        "audit",
        "rollout-inductor-update-eager",
    )
    # A training journal carries no configuration record at all.
    assert telemetry._configuration_context({}) == ("rollout", "eager")
