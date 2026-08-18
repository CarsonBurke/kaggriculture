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
    assert _scalars(log_dir / "rollout-compiled/games_16/parity/unit", "parity/kl") == [(0, 0.25)]
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
    assert _scalars(log_dir / "heads/unit", "holdout/accuracy") == [(0, 0.875), (1, 0.9375)]
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

    active = EventAccumulator(str(log_dir / "opponents/active")).Reload().Tags()["scalars"]
    assert set(active) == {
        "opponents/count",
        "opponents/games",
        "opponents/score_rate",
        "opponents/mean_margin",
        "opponents/score_rate_min",
        "opponents/score_rate_max",
    }
    # Rates are weighted by games played, so the opponent drawn for 24 games
    # counts for three times the one drawn for 8: (0.5*24 + 0.25*8)/32.
    assert _scalars(log_dir / "opponents/active", "opponents/score_rate")[0] == (1, 0.4375)
    assert _scalars(log_dir / "opponents/active", "opponents/mean_margin")[0] == (1, 62.5)
    assert _scalars(log_dir / "opponents/active", "opponents/count") == [(1, 2.0), (2, 1.0)]
    assert _scalars(log_dir / "opponents/active", "opponents/score_rate_min")[0] == (1, 0.25)
    assert _scalars(log_dir / "opponents/historical", "opponents/games") == [(1, 16.0)]

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

    builtin = log_dir / "opponents/builtin"
    assert _scalars(builtin, "opponents/count") == [(1, 2.0)]
    assert _scalars(builtin, "opponents/games") == [(1, 27.0)]
    # Both built-ins are present and weighted by games: (0.0*14 + 0.25*13)/27.
    assert _scalars(builtin, "opponents/score_rate")[0][1] == pytest.approx(0.25 * 13 / 27)
    assert _scalars(builtin, "opponents/score_rate_min") == [(1, 0.0)]
    assert _scalars(builtin, "opponents/score_rate_max") == [(1, 0.25)]
    # The snapshot key beside them, which has no underscore, is unaffected.
    assert _scalars(log_dir / "opponents/active", "opponents/games") == [(1, 14.0)]
    assert not (log_dir / "opponents/unclassified").exists()


def test_a_cohort_is_a_run_so_its_statistics_share_one_chart(tmp_path: Path) -> None:
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

    for run, expected in (("wave", 0.5), ("self-play", 0.25), ("league", 0.75)):
        assert _scalars(log_dir / run, "outcome/score_rate") == [(1, expected)]
    # A shipped unit action is filed by what it acts on, and an action the game
    # grows still lands on a chart of its own under the family fallback.
    assert _scalars(log_dir / "wave", "logistics/move_fraction") == [(1, 0.125)]
    assert _scalars(log_dir / "wave", "unit-actions/teleport_fraction") == [(1, 0.0625)]
    # The family token is absorbed by the category where it would only stutter,
    # and kept where the category holds more than one family.
    assert _scalars(log_dir / "league", "economy/money_median") == [(1, 0.0)]


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
            entropy_sums=np.zeros((trajectories, horizon), dtype=np.float32),
            elapsed_seconds=1.0,
        )
    )
    # Everything placed by a prefix rule rather than by the table: the parity
    # audit's per-head statistics and their per-head abort thresholds, and
    # behavior cloning's per-head holdout scores. These are the families the
    # first version of this helper still missed -- 26 parity fields and the
    # whole holdout set -- so adding two entries to `PARITY_STATISTICS` could
    # push every `parity/<head>` run over budget with both checks green.
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


def test_no_two_fields_share_one_run_and_tag() -> None:
    """A collision is silent: two series at the same step render as one line.

    Nothing raises, nothing is dropped, and the chart shows a plausible noisy
    curve that is actually two different measurements interleaved.
    """
    seen: dict[tuple[str, str], str] = {}
    for name in _mirrored_field_names():
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
    """
    benchmark_run = telemetry._BENCHMARK_BATCH_RUN.format(kind="ppo", mode="eager", games=112)
    counts: dict[tuple[str, str], int] = {}
    for name in _mirrored_field_names():
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
