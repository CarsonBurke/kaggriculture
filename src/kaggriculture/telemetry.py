"""Crash-recoverable JSONL journals mirrored into TensorBoard event logs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

#: Bumped whenever the *logic* that places journal fields changes. The layout
#: tables are not covered here; they are fingerprinted into the version below,
#: because a hand-maintained number fails silently -- someone renames a tag,
#: forgets the bump, and every existing mirror keeps serving the old scheme.
#: 4: benchmark mode became four-valued and gained an `audit` kind. The
#: fingerprint below covers the run and tag format strings, not the logic that
#: chooses what to substitute into them, so a mirror written under epoch 3
#: would keep serving `ppo-eager` for runs the new derivation calls
#: `update-only` and go unrebuilt -- exactly the silent failure this number
#: exists to prevent.
_LAYOUT_EPOCH = 4
_MANIFEST_NAME = ".kaggriculture-tensorboard.json"


class SummaryWriterLike(Protocol):
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None: ...

    def add_text(self, tag: str, text_string: str, global_step: int) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


WriterFactory = Callable[[Path], SummaryWriterLike]


@dataclass(frozen=True)
class JournalSnapshot:
    path: Path
    sha256: str
    size_bytes: int
    records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class MigrationResult:
    log_dir: Path
    records: int
    rebuilt: bool
    source_sha256: str


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant in metrics journal: {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key in metrics journal: {key}")
        result[key] = value
    return result


def _parse_journal_lines(
    contents: bytes, path: Path, first_line_number: int = 1
) -> tuple[dict[str, Any], ...]:
    """Strictly parse complete journal lines, ignoring only a torn final suffix."""
    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"metrics journal is not UTF-8: {path}") from error
    if text and not text.endswith("\n"):
        text = text.rpartition("\n")[0]
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=first_line_number):
        if not line:
            raise ValueError(f"metrics journal contains a blank record: {path}:{line_number}")
        try:
            record = json.loads(
                line,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_object_without_duplicate_keys,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid metrics record at {path}:{line_number}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"metrics record is not an object: {path}:{line_number}")
        records.append(record)
    return tuple(records)


def read_jsonl_snapshot(path: Path) -> JournalSnapshot:
    """Read one atomic journal snapshot, recovering only a torn final suffix."""
    path = Path(path).expanduser().resolve()
    contents = path.read_bytes() if path.exists() else b""
    digest = hashlib.sha256(contents).hexdigest()
    records = _parse_journal_lines(contents, path)
    return JournalSnapshot(path, digest, len(contents), records)


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _event_files(log_dir: Path) -> dict[str, dict[str, int | str]]:
    if not log_dir.exists():
        return {}
    result: dict[str, dict[str, int | str]] = {}
    for path in sorted(log_dir.rglob("events.out.tfevents.*")):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"TensorBoard event path is not a regular file: {path}")
        relative = path.relative_to(log_dir).as_posix()
        result[relative] = {"sha256": _file_sha256(path), "size_bytes": path.stat().st_size}
    return result


def _manifest_payload(snapshot: JournalSnapshot, log_dir: Path) -> dict[str, Any]:
    return {
        "format_version": TENSORBOARD_MIRROR_FORMAT_VERSION,
        "source": {
            "name": snapshot.path.name,
            "sha256": snapshot.sha256,
            "size_bytes": snapshot.size_bytes,
            "records": len(snapshot.records),
        },
        "event_files": _event_files(log_dir),
    }


def _write_manifest(snapshot: JournalSnapshot, log_dir: Path) -> None:
    _write_manifest_payload(_manifest_payload(snapshot, log_dir), log_dir)


def _write_manifest_payload(payload: dict[str, Any], log_dir: Path) -> None:
    destination = log_dir / _MANIFEST_NAME
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=log_dir
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _mirror_is_current(snapshot: JournalSnapshot, log_dir: Path) -> bool:
    manifest_path = log_dir / _MANIFEST_NAME
    if not log_dir.is_dir() or not manifest_path.is_file() or manifest_path.is_symlink():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = _manifest_payload(snapshot, log_dir)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return manifest == expected


def _manifest_event_files_match(log_dir: Path) -> bool:
    manifest_path = log_dir / _MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest.get("format_version") == TENSORBOARD_MIRROR_FORMAT_VERSION and manifest.get(
            "event_files"
        ) == _event_files(log_dir)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _require_owned_or_empty_directory(destination: Path) -> None:
    if not destination.exists():
        return
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError(f"TensorBoard destination is not a safe directory: {destination}")
    if not any(destination.iterdir()):
        return
    owner = destination / _MANIFEST_NAME
    if not owner.is_file() or owner.is_symlink():
        raise ValueError(
            "refusing to replace a nonempty directory not owned by the TensorBoard mirror: "
            f"{destination}"
        )


def _default_writer_factory(log_dir: Path) -> SummaryWriterLike:
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir)


class _RunWriters:
    """The writers of one mirror, one per TensorBoard run directory.

    TensorBoard takes a run to be a directory of event files and overlays equal
    tags across runs on a single chart, so a split that exists to be compared
    -- self-play against league, one replay-parity head against another -- is a
    directory here rather than a tag prefix. Runs open on first use because
    which ones a mirror needs is a property of the records it is given, not of
    the mirror: a league category that appears at iteration 300 opens its run
    then.
    """

    def __init__(self, root: Path, factory: WriterFactory) -> None:
        self._root = root
        self._factory = factory
        self._writers: dict[str, SummaryWriterLike] = {}

    def writer(self, run: str = "") -> SummaryWriterLike:
        existing = self._writers.get(run)
        if existing is not None:
            return existing
        directory = self._root / run if run else self._root
        directory.mkdir(parents=True, exist_ok=True)
        created = self._factory(directory)
        self._writers[run] = created
        return created

    def add_scalar(self, run: str, tag: str, value: float, step: int) -> None:
        self.writer(run).add_scalar(tag, value, step)

    def flush(self) -> None:
        for writer in self._writers.values():
            writer.flush()

    def close(self) -> None:
        # Every writer is closed even if one raises, because a writer left open
        # holds an event file the mirror is about to move or replace.
        errors: list[BaseException] = []
        for writer in self._writers.values():
            try:
                writer.close()
            except BaseException as error:
                errors.append(error)
        self._writers.clear()
        if errors:
            raise errors[0]


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    converted = float(value)
    if not math.isfinite(converted):
        raise FloatingPointError(f"non-finite TensorBoard scalar: {value}")
    return converted


#: Step fields a training journal may be keyed by, in precedence order. PPO
#: counts iterations and behavior cloning counts epochs; both are a single
#: monotonic step index over a flat record of scalars, and neither carries the
#: `event` tag that marks a benchmark record.
_TRAINING_STEP_FIELDS = ("iteration", "epoch")


def training_step_field(record: dict[str, Any]) -> str | None:
    """The step field of a training journal record, or None if it is not one.

    A record that reaches TensorBoard without matching here is mirrored as a
    JSON text blob rather than as scalars, which is the right fallback for
    benchmark metadata and useless for a training curve -- so every training
    journal has to be recognized here or its curves silently do not appear.
    """
    if record.get("event") is not None:
        return None
    return next(
        (name for name in _TRAINING_STEP_FIELDS if type(record.get(name)) is int),
        None,
    )


_COMPILE_MODES = {
    (False, False): "eager",
    (True, False): "rollout-only",
    (False, True): "update-only",
    (True, True): "compiled",
}


def _configuration_context(configuration: dict[str, Any]) -> tuple[str, str]:
    # Three producers write configuration records, and each names its
    # compilation differently because each compiles different things. Reading
    # one key for all of them labels the others "eager" whatever they measured.
    #
    #   ppo      the iteration benchmark: a rollout phase and an update phase,
    #            decided independently. `compile_rollout` sits at top level and
    #            `compile_update` is nested under `ppo` -- a schema asymmetry,
    #            so both have to be read from where they actually are.
    #   audit    the replay-parity audit: both knobs at top level, and no
    #            `self_play_game_counts` (it takes a single `self_play_games`),
    #            which is why it must be discriminated explicitly rather than
    #            falling through to the rollout branch and reading a key it
    #            never writes.
    #   rollout  the rust rollout benchmark: only a collector, so the single
    #            `compile_models` is the whole of its mode.
    #
    # The mode is four-valued rather than a boolean because the mixed
    # configurations are real: production runs an eager collector against a
    # compiled update, and calling that "eager" would put a run whose update
    # phase is ~1.8x faster in the same series as a true eager one, where the
    # difference reads as hardware noise instead of as the knob it is.
    #
    # Absent means off throughout, because this is also reached with no
    # configuration record at all -- a training journal has none, and
    # `_benchmark_context` falls back to an empty mapping rather than failing a
    # mirror over a label.
    if "self_play_game_counts" in configuration:
        kind = "ppo"
        ppo = configuration.get("ppo")
        knobs = (
            configuration.get("compile_rollout") is True,
            isinstance(ppo, dict) and ppo.get("compile_update") is True,
        )
    elif "self_play_games" in configuration:
        kind = "audit"
        knobs = (
            configuration.get("compile_rollout") is True,
            configuration.get("compile_update") is True,
        )
    else:
        kind = "rollout"
        collector = configuration.get("compile_models") is True
        return kind, "compiled" if collector else "eager"
    return kind, _COMPILE_MODES[knobs]


def _benchmark_context(records: tuple[dict[str, Any], ...]) -> tuple[str, str]:
    configuration = next(
        (record for record in records if record.get("event") == "configuration"),
        {},
    )
    return _configuration_context(configuration)


#: Behavioral statistics reach the journal three times over: once for the whole
#: mixed wave, once for its self-play half under a `self_play_` prefix, and once
#: for its league half under a `league_` prefix. Left as tags that is one chart
#: drawn three times in three categories; made TensorBoard *runs* it is one
#: chart carrying three series, which is the comparison the three exist to
#: support. The empty prefix is last because it matches everything.
_COHORT_RUNS: tuple[tuple[str, str], ...] = (
    ("self_play_", "self-play"),
    ("league_", "league"),
    ("", "wave"),
)

#: Category by the leading family token of a behavioral statistic, after its
#: cohort prefix is removed, and whether the category name already says what
#: that token says. Families are open sets -- the game grows unit actions and
#: market kinds -- so they are matched by prefix rather than enumerated, and a
#: new action appears as a new chart in the right category instead of as a new
#: category. The flag drops the token from the chart name where keeping it only
#: stutters (`unit-actions/unit_pass_fraction`), and keeps it where the
#: category holds more than one family: `economy` carries margins as well as
#: money, so `economy/money_median` still has to say which.
#:
#: Every unit action shipped today is placed explicitly below, so `unit-actions`
#: now collects only actions the game has grown since. That is the useful
#: behavior for an open set: a new action is charted immediately, in a category
#: whose name says it has not been filed yet.
_BEHAVIOR_FAMILIES: tuple[tuple[str, str, bool], ...] = (
    ("unit_", "unit-actions", True),
    ("market_", "market-actions", True),
    ("money_", "economy", False),
    ("rollout_", "rollout", True),
)

#: Placement, as (category, chart), for the behavioral statistics a family
#: cannot place: those whose leading token is not a family at all, and those
#: whose family has outgrown one category. The thirteen unit actions were one
#: `unit-actions` accordion, which is past the point where a reader can find the
#: one that moved, so they are split by what the unit acts on -- the crop cycle,
#: the animals, and everything else the unit does with itself and its cargo. The
#: split is written out because it is a fact about the game, not about the
#: names: `build` raises a coop or a pasture and belongs with the animals, and
#: `shed` is the whole DROP..PICKUP_SHEEP range -- one deposit against every
#: withdrawal size for every good -- so it counts transfers at the shed, not
#: time spent carrying, and its chart says so.
_BEHAVIOR_CATEGORIES: dict[str, tuple[str, str]] = {
    "score_rate": ("outcome", "score_rate"),
    "seat_zero_score_rate": ("outcome", "seat_zero_score_rate"),
    "seat_one_score_rate": ("outcome", "seat_one_score_rate"),
    "tie_fraction": ("outcome", "tie_fraction"),
    "margin_abs_mean": ("economy", "margin_abs_mean"),
    # How much is traded rather than which trade was chosen, so these sit with
    # the money they move instead of with the market kinds. Moving them also
    # leaves `market-actions` room: it is a prefix-matched open set that was
    # sitting exactly at the budget, and a market kind added to the game would
    # have pushed it over the same way the unit actions went over.
    "market_quantity_fraction": ("economy", "trade_quantity_fraction"),
    "market_quantity_mean": ("economy", "trade_quantity_mean"),
    "unit_dig_fraction": ("crop-actions", "dig_fraction"),
    "unit_plant_fraction": ("crop-actions", "plant_fraction"),
    "unit_water_fraction": ("crop-actions", "water_fraction"),
    "unit_fertilize_fraction": ("crop-actions", "fertilize_fraction"),
    "unit_harvest_fraction": ("crop-actions", "harvest_fraction"),
    "unit_build_fraction": ("livestock-actions", "build_fraction"),
    "unit_place_fraction": ("livestock-actions", "place_fraction"),
    "unit_feed_fraction": ("livestock-actions", "feed_fraction"),
    "unit_care_fraction": ("livestock-actions", "care_fraction"),
    "unit_collect_fertilizer_fraction": ("livestock-actions", "collect_fertilizer_fraction"),
    "unit_pass_fraction": ("logistics", "pass_fraction"),
    "unit_move_fraction": ("logistics", "move_fraction"),
    "unit_shed_fraction": ("logistics", "shed_transfer_fraction"),
}

#: Full tag for each field a training record carries about the update itself.
#: Written out rather than derived from the field name because the grouping is
#: a judgement about what belongs on screen together, which no naming
#: convention in the journal encodes -- `entropy` and `rollout_entropy` measure
#: different things, and `actor_learning_rate` belongs with the other
#: schedules rather than with the actor's losses.
_TRAINING_TAGS = {
    "policy_loss": "actor/policy_loss",
    "entropy": "actor/entropy",
    "actor_updates": "actor/updates",
    "actor_gradient_norm": "actor/gradient_norm",
    "advantage_mean": "actor/advantage_mean",
    "advantage_std": "actor/advantage_std",
    # What bounds the step, separated from what takes it. These four are read
    # together and against each other -- a clip fraction climbing while the KL
    # stays flat is a different situation from both climbing -- and they were
    # the four that pushed `actor` past the point of being scannable.
    "approx_kl": "trust-region/approx_kl",
    "max_approx_kl": "trust-region/max_approx_kl",
    "clip_fraction": "trust-region/clip_fraction",
    "kl_early_stop": "trust-region/kl_early_stop",
    # `critic` is how well the critic is fitting; `value` is what it and its
    # targets actually look like. Splitting them keeps either category readable
    # at a glance and keeps both inside the ten-chart budget, which one combined
    # category of twelve would not be.
    "value_loss": "critic/value_loss",
    "value_loss_first_epoch": "critic/value_loss_first_epoch",
    "value_loss_last_epoch": "critic/value_loss_last_epoch",
    "critic_fit_explained_variance_first_epoch": "critic/fit_explained_variance_first_epoch",
    "critic_fit_explained_variance_last_epoch": "critic/fit_explained_variance_last_epoch",
    "lambda_return_explained_variance": "critic/lambda_return_explained_variance",
    "monte_carlo_explained_variance": "critic/monte_carlo_explained_variance",
    "value_target_correlation": "critic/target_correlation",
    "critic_gradient_norm": "critic/gradient_norm",
    "value_target_mean": "value/target_mean",
    "value_target_std": "value/target_std",
    "value_target_min": "value/target_min",
    "value_target_max": "value/target_max",
    "value_target_saturated_fraction": "value/target_saturated_fraction",
    "value_prediction_mean": "value/prediction_mean",
    "value_prediction_std": "value/prediction_std",
    "actor_learning_rate": "schedule/actor_learning_rate",
    "critic_learning_rate": "schedule/critic_learning_rate",
    "actor_gae_lambda": "schedule/actor_gae_lambda",
    "gamma": "schedule/gamma",
    "epochs": "schedule/epochs",
    "updates": "schedule/updates",
    # `timing` is how long a phase took and `throughput` is how much it got
    # done per unit of that time. Kept apart because the benchmark journal adds
    # four more of each to the same names, and one combined category would be
    # over budget there while reading no better here.
    "iteration_seconds": "timing/iteration_seconds",
    "rollout_seconds": "timing/rollout_seconds",
    "update_seconds": "timing/update_seconds",
    "total_seconds": "timing/total_seconds",
    "elapsed_seconds": "timing/elapsed_seconds",
    "elapsed_hours": "timing/elapsed_hours",
    "update_replay_parity_seconds": "timing/replay_parity_seconds",
    "rollout_states_per_second": "throughput/rollout_states_per_second",
    "critic_replayed_states_per_second": "throughput/critic_replayed_states_per_second",
    "learner_states_per_rollout_second": "throughput/learner_states_per_rollout_second",
    "physical_games_per_rollout_second": "throughput/physical_games_per_rollout_second",
    "iterations_per_hour": "throughput/iterations_per_hour",
    "states": "throughput/states",
    # What one benchmark iteration was asked to do, and what it cost the
    # machine to do it. Neither is a curve over training -- the benchmark holds
    # them fixed per batch -- but both are what a batch-size sweep is read for.
    "league_games": "batch/league_games",
    "physical_games": "batch/physical_games",
    "learner_states": "batch/learner_states",
    "learner_trajectories": "batch/learner_trajectories",
    "actor_parameters": "capacity/actor_parameters",
    "critic_parameters": "capacity/critic_parameters",
    "peak_cuda_bytes": "capacity/peak_cuda_bytes",
    "process_lifetime_max_rss_kib": "capacity/process_lifetime_max_rss_kib",
    # A benchmark batch summary reduces its repeats to a cold reading and a
    # steady median. They are two answers to the same questions, so they are two
    # categories rather than one of eight interleaved names.
    "cold_total_seconds": "cold/total_seconds",
    "cold_iterations_per_hour": "cold/iterations_per_hour",
    "cold_physical_games_per_rollout_second": "cold/physical_games_per_rollout_second",
    "steady_total_seconds_median": "steady/total_seconds_median",
    "steady_iterations_per_hour_median": "steady/iterations_per_hour_median",
    "steady_physical_games_per_rollout_second_median": (
        "steady/physical_games_per_rollout_second_median"
    ),
    "steady_critic_replayed_states_per_second_median": (
        "steady/critic_replayed_states_per_second_median"
    ),
    "replay_parity_breached": "parity/breached",
    # The live per-iteration reading of the quantity the audit below samples on
    # a cadence. It is a numerics measurement, not a trust-region one -- the
    # trust region never sees it, because the update replays its own behavior
    # likelihoods -- so it belongs beside the audit rather than beside the KL.
    "first_minibatch_approx_kl": "parity/first_minibatch_kl",
    # Placed explicitly so the per-head prefix rule does not route it into the
    # `parity/<head>` runs, where it would share a chart with the per-head and
    # joint sampling-versus-replay divergences. It measures the replay against
    # the update forward instead, so overlaying it on those would invite exactly
    # the comparison that mis-set its bound.
    "update_replay_first_minibatch_kl": "parity/replay_to_update_kl",
    "update_replay_mean_minibatch_kl": "parity/replay_to_update_mean_kl",
    # Same hazard, and the one the per-head rule actually fell into: `max`,
    # `joint` and `minibatch` are not heads, but the rule splits on the first
    # token and made each of them a `parity/<head>` run. That put six series on
    # `parity/kl`, three of them functions of the other three -- `max` is the
    # maximum over the heads and `joint` their component-weighted mean -- so the
    # one reading the chart exists for, a single head departing from the others,
    # was drawn over by two curves that follow the departing head by definition.
    "update_replay_max_kl": "parity/max_kl",
    "update_replay_joint_kl": "parity/joint_kl",
    "update_replay_minibatch_kl": "parity/minibatch_kl",
    "update_replay_max_tail_fraction": "parity/max_tail_fraction",
    "update_replay_max_ratio_error": "parity/max_ratio_error",
    # Behavior cloning writes its own journal through the same mirror. Its
    # fields are few and unprefixed, so they are placed by name into the same
    # categories the RL run uses -- a learning rate is a schedule in both.
    "train_loss": "loss/train",
    "holdout_nll": "loss/holdout_nll",
    "learning_rate": "schedule/learning_rate",
    "seconds": "throughput/seconds",
}

#: Fields that are bookkeeping rather than a curve. A resume token plotted over
#: time says nothing about training and costs a chart to say it.
_UNPLOTTED = frozenset(("next_seed",))

#: Statistics measured once per policy head, as (field prefix, run directory,
#: tag category). The head becomes a run so that unit, kind and quantity land
#: as three series on one chart, which is how they are actually read -- a
#: replay-parity breach is recognized by one head departing from the others,
#: and that comparison is invisible across three separate charts.
_PER_HEAD_PREFIXES: tuple[tuple[str, str, str], ...] = (
    ("update_replay_", "parity", "parity"),
    ("holdout_", "heads", "holdout"),
)

_OPPONENT_PREFIX = "league_opponent_"

#: The run and tags the aggregated league-opponent curves are written at, and
#: the per-opponent fields reduced into them. Hoisted out of
#: `_opponent_scalars` so `_layout_fingerprint` below can cover them. While they
#: were literals inside that function the fingerprint did not see them, so
#: renaming `opponents/score_rate` left every mirror serving two schemes at once
#: with the format version unchanged -- the exact failure the fingerprint exists
#: to prevent, in the one part of the namespace it did not reach.
_OPPONENT_RUN_ROOT = "opponents"
_OPPONENT_CATEGORY = "opponents"
_OPPONENT_AGGREGATE_FIELDS = ("score_rate", "mean_margin")
_OPPONENT_UNCLASSIFIED = "unclassified"

#: Run layout for a benchmark journal, which is keyed by batch size and repeat
#: rather than by a monotonic step. Same reason as above: these decide where
#: every benchmark scalar lands, so the identity has to include them.
_BENCHMARK_BATCH_RUN = "{kind}-{mode}/games_{games}"
_BENCHMARK_SUMMARY_RUN = "{kind}-{mode}/batch_summary"
_BENCHMARK_METADATA_TAG = "{kind}-{mode}/metadata/{event}"

#: Anything the rules above do not recognize. It is a visible category rather
#: than a silent drop so that a metric added to the journal shows up somewhere
#: and can be filed deliberately, instead of being absent from TensorBoard with
#: nothing to indicate it was ever written.
_UNCATEGORIZED = "misc"


def _layout_fingerprint() -> str:
    """Identify the mirror layout by the tables that define it.

    A mirror is only readable under the layout that wrote it, and mirrors are
    regenerated from the journal, so invalidating them costs a few seconds of
    replay. Deriving the identity from the tables means a renamed tag or a new
    category cannot leave a directory serving a mixture of two schemes.
    """
    payload = json.dumps(
        {
            "training": _TRAINING_TAGS,
            "cohorts": _COHORT_RUNS,
            "families": _BEHAVIOR_FAMILIES,
            "behavior": _BEHAVIOR_CATEGORIES,
            "per_head": _PER_HEAD_PREFIXES,
            "unplotted": sorted(_UNPLOTTED),
            "opponents": {
                "prefix": _OPPONENT_PREFIX,
                "run_root": _OPPONENT_RUN_ROOT,
                "category": _OPPONENT_CATEGORY,
                "aggregates": _OPPONENT_AGGREGATE_FIELDS,
                "unclassified": _OPPONENT_UNCLASSIFIED,
            },
            "benchmark": {
                "batch_run": _BENCHMARK_BATCH_RUN,
                "summary_run": _BENCHMARK_SUMMARY_RUN,
                "metadata_tag": _BENCHMARK_METADATA_TAG,
            },
            "uncategorized": _UNCATEGORIZED,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


#: The identity a mirror is checked against. Both the standalone staleness check
#: and the live mirror's manifest compare this, so a directory written under any
#: other layout is rebuilt in full rather than appended to.
TENSORBOARD_MIRROR_FORMAT_VERSION = f"{_LAYOUT_EPOCH}-{_layout_fingerprint()}"


def _placement(name: str) -> tuple[str, str] | None:
    """Return the (run, tag) a training-journal field is mirrored at, or None."""
    if name in _UNPLOTTED:
        return None
    tag = _TRAINING_TAGS.get(name)
    if tag is not None:
        return "", tag
    for prefix, run_root, category in _PER_HEAD_PREFIXES:
        if not name.startswith(prefix):
            continue
        head, _, statistic = name[len(prefix) :].partition("_")
        if statistic:
            return f"{run_root}/{head}", f"{category}/{statistic}"
    for prefix, run in _COHORT_RUNS:
        if not name.startswith(prefix):
            continue
        statistic = name[len(prefix) :]
        placement = _BEHAVIOR_CATEGORIES.get(statistic)
        if placement is not None:
            category, chart = placement
            return run, f"{category}/{chart}"
        for family, category, absorbs in _BEHAVIOR_FAMILIES:
            if statistic.startswith(family):
                # A statistic named after the bare family has nothing left once
                # the family is dropped, and a tag ending in a slash is not a
                # chart, so it keeps its name.
                chart = statistic[len(family) :] if absorbs else statistic
                return run, f"{category}/{chart or statistic}"
    return "", f"{_UNCATEGORIZED}/{name}"


def _opponent_scalars(record: dict[str, Any]) -> Iterator[tuple[str, str, float]]:
    """Yield league-opponent statistics aggregated over opponent categories.

    Per-opponent fields are keyed by the opponent's checkpoint index, so
    mirroring them field by field mints new tags for every opponent the league
    ever samples and the namespace grows without bound -- 494 tags by the point
    this was reported, each a curve defined only on the iterations that one
    opponent happened to be drawn. The journal keeps the per-opponent detail
    for post-hoc league analysis; what a *curve* can honestly show is the
    aggregate over a category, which is defined on every iteration.

    Rates and margins are weighted by games played rather than averaged over
    opponents, so an opponent drawn for two games does not count as much as one
    drawn for twenty.
    """
    opponents: dict[str, dict[str, Any]] = {}
    for name, value in record.items():
        if not name.startswith(_OPPONENT_PREFIX):
            continue
        identifier, _, field = name[len(_OPPONENT_PREFIX) :].partition("_")
        if field:
            opponents.setdefault(identifier, {})[field] = value

    categories: dict[str, list[dict[str, Any]]] = {}
    for opponent in opponents.values():
        category = opponent.get("category")
        key = category if isinstance(category, str) and category else _OPPONENT_UNCLASSIFIED
        categories.setdefault(key, []).append(opponent)

    for category, members in sorted(categories.items()):
        run = f"{_OPPONENT_RUN_ROOT}/{category}"
        games = [max(_number(member.get("games")) or 0.0, 0.0) for member in members]
        yield run, f"{_OPPONENT_CATEGORY}/count", float(len(members))
        yield run, f"{_OPPONENT_CATEGORY}/games", sum(games)
        for field in _OPPONENT_AGGREGATE_FIELDS:
            weighted = [
                (value, weight)
                for member, weight in zip(members, games, strict=True)
                if (value := _number(member.get(field))) is not None
            ]
            if not weighted:
                continue
            total = sum(weight for _, weight in weighted)
            # An iteration in which every opponent of a category played zero
            # games still has a defined membership, so it falls back to the
            # unweighted mean rather than dividing by zero or vanishing.
            yield (
                run,
                f"{_OPPONENT_CATEGORY}/{field}",
                sum(value * weight for value, weight in weighted) / total
                if total > 0
                else sum(value for value, _ in weighted) / len(weighted),
            )
        # `is not None` rather than truthiness: a score rate of exactly zero is
        # a measurement, and losing every game is precisely when the spread
        # across a category is worth seeing.
        rates = [
            rate for member in members if (rate := _number(member.get("score_rate"))) is not None
        ]
        if rates:
            yield run, f"{_OPPONENT_CATEGORY}/score_rate_min", min(rates)
            yield run, f"{_OPPONENT_CATEGORY}/score_rate_max", max(rates)


def _write_record(
    writers: _RunWriters,
    record: dict[str, Any],
    record_index: int,
    context: tuple[str, str],
) -> None:
    step_field = training_step_field(record)
    if step_field is not None:
        step = int(record[step_field])
        for name, value in record.items():
            if name == step_field or name.startswith(_OPPONENT_PREFIX):
                continue
            scalar = _number(value)
            placement = None if scalar is None else _placement(name)
            if scalar is not None and placement is not None:
                writers.add_scalar(placement[0], placement[1], scalar, step)
        for run, tag, aggregate in _opponent_scalars(record):
            writers.add_scalar(run, tag, aggregate, step)
        return

    event = record.get("event")
    kind, mode = context
    if event in {"iteration", "repeat"}:
        games = record.get("self_play_games", record.get("games"))
        step = record.get("repeat", record_index)
        if type(games) is not int or type(step) is not int:
            raise ValueError(f"benchmark {event} record lacks integer games/repeat fields")
        run_root = _BENCHMARK_BATCH_RUN.format(kind=kind, mode=mode, games=games)
        ignored = {"event", "repeat", "games", "self_play_games"}
    elif event == "batch_summary":
        games = record.get("self_play_games", record.get("games"))
        if type(games) is not int:
            raise ValueError("benchmark batch summary lacks an integer game count")
        step = games
        run_root = _BENCHMARK_SUMMARY_RUN.format(kind=kind, mode=mode)
        ignored = {"event", "games", "self_play_games"}
    else:
        writers.writer().add_text(
            _BENCHMARK_METADATA_TAG.format(kind=kind, mode=mode, event=event or "record"),
            "```json\n" + json.dumps(record, indent=2, sort_keys=True) + "\n```",
            record_index,
        )
        return

    # The batch size is a run and the metric keeps the category it has in a
    # training journal, because a benchmark record *is* a training iteration --
    # 56 of its scalars are the same fields under the same names. Flattening it
    # into `{kind}/{mode}/games_{n}/{name}` tags instead put every one of those
    # under a single first path component, so the whole benchmark mirror opened
    # as one 63-chart accordion. As runs they collapse onto the training layout,
    # and the sweep reads the way it is meant to: one chart per metric with one
    # series per batch size, which tags in separate categories cannot show.
    for name, value in record.items():
        scalar = _number(value)
        if name in ignored or scalar is None:
            continue
        placement = _placement(name)
        if placement is None:
            continue
        run, tag = placement
        writers.add_scalar(f"{run_root}/{run}" if run else run_root, tag, scalar, step)


def _write_records(writers: _RunWriters, snapshot: JournalSnapshot) -> None:
    context = _benchmark_context(snapshot.records)
    for index, record in enumerate(snapshot.records):
        _write_record(writers, record, index, context)
    writers.flush()


def _replace_derived_directory(staging: Path, destination: Path) -> None:
    _require_owned_or_empty_directory(destination)
    backup: Path | None = None
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"TensorBoard destination is not a directory: {destination}")
        backup = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.previous.", dir=destination.parent)
        )
        backup.rmdir()
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(backup)


def migrate_jsonl_to_tensorboard(
    journal_path: Path,
    log_dir: Path,
    *,
    force: bool = False,
    allow_missing_journal: bool = False,
    writer_factory: WriterFactory = _default_writer_factory,
) -> MigrationResult:
    """Idempotently rebuild a TensorBoard mirror from canonical JSONL bytes."""
    selected_journal = Path(journal_path).expanduser()
    if not selected_journal.exists() and not allow_missing_journal:
        raise FileNotFoundError(selected_journal)
    snapshot = read_jsonl_snapshot(selected_journal)
    selected_destination = Path(log_dir).expanduser()
    if selected_destination.is_symlink():
        raise ValueError(f"TensorBoard destination cannot be a symlink: {selected_destination}")
    destination = selected_destination.resolve()
    if destination.parent == destination or destination in snapshot.path.parents:
        raise ValueError("TensorBoard destination cannot contain the source journal")
    _require_owned_or_empty_directory(destination)
    if not force and _mirror_is_current(snapshot, destination):
        return MigrationResult(destination, len(snapshot.records), False, snapshot.sha256)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    installed = False
    try:
        writers = _RunWriters(staging, writer_factory)
        try:
            _write_records(writers, snapshot)
        finally:
            writers.close()
        _write_manifest(snapshot, staging)
        _replace_derived_directory(staging, destination)
        installed = True
    finally:
        if not installed and staging.exists():
            shutil.rmtree(staging)
    return MigrationResult(destination, len(snapshot.records), True, snapshot.sha256)


class TensorboardMirror:
    """Live TensorBoard writer that repairs itself from its JSONL source journal.

    The journal and TensorBoard event files are both append-only, so per-record
    integrity tracking reads and hashes only the appended bytes. Any anomaly
    (shrinkage, torn suffix, parse failure, manifest mismatch) falls back to
    the full-read verification and rebuild paths.
    """

    def __init__(
        self,
        journal_path: Path,
        log_dir: Path,
        *,
        writer_factory: WriterFactory = _default_writer_factory,
    ) -> None:
        self.journal_path = Path(journal_path).expanduser().resolve()
        selected_log_dir = Path(log_dir).expanduser()
        self.writer_factory = writer_factory
        migrate_jsonl_to_tensorboard(
            self.journal_path,
            selected_log_dir,
            allow_missing_journal=True,
            writer_factory=self.writer_factory,
        )
        self.log_dir = selected_log_dir.resolve()
        self._open_writer()

    def _reset_journal_state(self) -> None:
        self._journal_sha = hashlib.sha256()
        self._journal_size = 0
        self._journal_records = 0
        self._last_record: dict[str, Any] | None = None
        self._configuration: dict[str, Any] = {}
        self._event_hashes: dict[str, tuple[Any, int]] = {}

    def _absorb_journal_records(self, records: tuple[dict[str, Any], ...]) -> None:
        self._journal_records += len(records)
        if records:
            self._last_record = records[-1]
        if not self._configuration:
            self._configuration = next(
                (record for record in records if record.get("event") == "configuration"),
                {},
            )

    def _reload_journal_state(self) -> None:
        self._reset_journal_state()
        contents = self.journal_path.read_bytes() if self.journal_path.exists() else b""
        if not contents.endswith(b"\n"):
            # Track only the newline-terminated prefix. A torn suffix may later
            # be truncated and replaced by an append of identical length, which
            # byte size alone cannot distinguish from an unchanged file.
            cut = contents.rfind(b"\n")
            contents = contents[: cut + 1] if cut >= 0 else b""
        records = _parse_journal_lines(contents, self.journal_path)
        self._journal_sha.update(contents)
        self._journal_size = len(contents)
        self._absorb_journal_records(records)

    def _extend_journal_state(self) -> bool:
        """Absorb appended journal bytes; False demands a full state reload."""
        size = self.journal_path.stat().st_size if self.journal_path.exists() else 0
        if size < self._journal_size:
            return False
        if size == self._journal_size:
            return True
        with self.journal_path.open("rb") as stream:
            stream.seek(self._journal_size)
            appended = stream.read(size - self._journal_size)
        if len(appended) != size - self._journal_size or not appended.endswith(b"\n"):
            return False
        try:
            records = _parse_journal_lines(
                appended, self.journal_path, first_line_number=self._journal_records + 1
            )
        except ValueError:
            return False
        self._journal_sha.update(appended)
        self._journal_size = size
        self._absorb_journal_records(records)
        return True

    def _current_event_files(self) -> dict[str, dict[str, int | str]]:
        """Hash event files incrementally, rehashing only on shrinkage."""
        result: dict[str, dict[str, int | str]] = {}
        for path in sorted(self.log_dir.rglob("events.out.tfevents.*")):
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"TensorBoard event path is not a regular file: {path}")
            relative = path.relative_to(self.log_dir).as_posix()
            cached = self._event_hashes.get(relative)
            size = path.stat().st_size
            if cached is None or size < cached[1]:
                hasher, offset = hashlib.sha256(), 0
            else:
                hasher, offset = cached
            if size > offset:
                with path.open("rb") as stream:
                    stream.seek(offset)
                    while chunk := stream.read(1 << 20):
                        hasher.update(chunk)
                        offset += len(chunk)
            self._event_hashes[relative] = (hasher, offset)
            result[relative] = {"sha256": hasher.hexdigest(), "size_bytes": offset}
        return result

    def _manifest_matches(self) -> bool:
        manifest_path = self.log_dir / _MANIFEST_NAME
        if not manifest_path.is_file() or manifest_path.is_symlink():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return (
                manifest.get("format_version") == TENSORBOARD_MIRROR_FORMAT_VERSION
                and manifest.get("event_files") == self._current_event_files()
            )
        except (OSError, ValueError, json.JSONDecodeError):
            return False

    def _open_writer(self) -> None:
        writers = _RunWriters(self.log_dir, self.writer_factory)
        try:
            # The root run opens eagerly so the mirror owns an event file from
            # the moment it exists, which is the condition the ownership check
            # and the manifest are both written against. Every other run opens
            # when a record first needs it.
            writers.writer().flush()
            _write_manifest(read_jsonl_snapshot(self.journal_path), self.log_dir)
        except BaseException:
            with suppress(Exception):
                writers.close()
            raise
        self.writers = writers
        self._reload_journal_state()

    def _repair(self) -> None:
        with suppress(Exception):
            self.writers.close()
        migrate_jsonl_to_tensorboard(
            self.journal_path,
            self.log_dir,
            force=True,
            allow_missing_journal=True,
            writer_factory=self.writer_factory,
        )
        self._open_writer()

    def record(self, payload: dict[str, Any]) -> None:
        """Mirror the journal's newly committed final record, repairing on failure."""
        if not self._extend_journal_state():
            self._reload_journal_state()
        if not self._journal_records or self._last_record != payload:
            raise ValueError("TensorBoard payload is not the committed final JSONL record")
        if not self._manifest_matches():
            self._repair()
            return
        try:
            context = _configuration_context(self._configuration)
            _write_record(self.writers, payload, self._journal_records - 1, context)
            self.writers.flush()
            _write_manifest_payload(
                {
                    "format_version": TENSORBOARD_MIRROR_FORMAT_VERSION,
                    "source": {
                        "name": self.journal_path.name,
                        "sha256": self._journal_sha.hexdigest(),
                        "size_bytes": self._journal_size,
                        "records": self._journal_records,
                    },
                    "event_files": self._current_event_files(),
                },
                self.log_dir,
            )
        except Exception:
            self._repair()

    def close(self) -> None:
        try:
            if not _manifest_event_files_match(self.log_dir):
                self._repair()
            self.writers.flush()
            self.writers.close()
            snapshot = read_jsonl_snapshot(self.journal_path)
            _write_manifest(snapshot, self.log_dir)
        except Exception:
            with suppress(Exception):
                self.writers.close()
            migrate_jsonl_to_tensorboard(
                self.journal_path,
                self.log_dir,
                force=True,
                allow_missing_journal=True,
                writer_factory=self.writer_factory,
            )
