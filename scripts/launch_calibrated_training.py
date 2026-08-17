#!/usr/bin/env python3
"""Launch production PPO with compilation selected by matched benchmark evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import torch

from kaggriculture.ppo import (
    MAX_FIRST_MINIBATCH_KL,
    MAX_UPDATE_REPLAY_KL,
    MAX_UPDATE_REPLAY_TAIL_FRACTION,
    MAX_VALUE_TARGET_SATURATED_FRACTION,
    UPDATE_COMPILE_MODES,
)
from kaggriculture.production import (
    PRODUCTION_EPISODE_STEPS,
    PRODUCTION_LEAGUE_ACTIVE_OPPONENTS,
    PRODUCTION_LEAGUE_GAMES,
    PRODUCTION_LEAGUE_HISTORICAL_OPPONENTS,
    PRODUCTION_OPPONENT_TEMPERATURE,
    PRODUCTION_ROLLOUT_BFLOAT16,
    PRODUCTION_SELF_PLAY_GAMES,
    PRODUCTION_TEMPERATURE,
    build_training_command,
    production_model_config,
    production_ppo_config,
    require_repository_launcher,
    resolve_resume_checkpoint,
)
from kaggriculture.provenance import (
    CALIBRATION_KNOBS,
    MINIMUM_COMPILE_SPEEDUP,
    ROLLOUT_FORWARD_MODE_KNOB,
    UNCOMPILED_CALIBRATION_MODES,
    UPDATE_COMPILE_MODE_KNOB,
    source_identity,
    validate_source_identity,
)
from kaggriculture.rollout import ROLLOUT_FORWARD_MODES

#: Steady iterations each report must contain for a median to mean anything.
#: The old floor was two repeats, which drops the cold start and leaves exactly
#: one steady iteration -- so every "median" was a single sample, and a knob
#: could be decided by one clock-boost dip or one noisy neighbour on the GPU.
#: That decision is then irreversible for the run: it is stamped into run
#: provenance and train_ppo refuses to resume under a different compile mode.
#: Five is what the eager side measured as stable (the steady update within
#: 0.15%, the steady total median within 0.24% of the two-repeat value), and
#: the compiled side needs at least as many because it has warmup eager does
#: not.
MINIMUM_STEADY_SAMPLES = 5
MINIMUM_CALIBRATION_REPEATS = MINIMUM_STEADY_SAMPLES + 1

#: The steady phase median each knob is expected to move, and so the one term
#: of the iteration budget its attribution swaps. The knob names themselves come
#: from `provenance.CALIBRATION_KNOBS`, which is the schema the decision this
#: writes is validated against; naming them twice is how the two drift apart.
KNOB_PHASE_MEDIANS = {
    ROLLOUT_FORWARD_MODE_KNOB: "steady_rollout_seconds_median",
    UPDATE_COMPILE_MODE_KNOB: "steady_update_seconds_median",
}
# What each knob reads as while its phase is uncompiled -- what the chain's
# all-eager first node must declare, and what a knob this decision leaves off
# keeps -- is `provenance.UNCOMPILED_CALIBRATION_MODES`, imported rather than
# restated here for the reason the knob names are: this launcher writes the
# decision that module validates. Neither knob's off value is `False`, because
# both knobs are mode-valued: `eager` is one of `ROLLOUT_FORWARD_MODES` and one
# of `UPDATE_COMPILE_MODES`, an execution mode like the others rather than the
# absence of a choice. The measured ranking is why on the collection side --
# isolated forward, median of 60: eager fp32 4.907 ms, cudagraphs fp32
# 5.309 ms, inductor fp32 2.720 ms, inductor bf16 1.626 ms -- so a boolean
# whose "on" value is `cudagraphs` selects the one mode slower than eager,
# which is what this launcher must not be able to certify.

#: Every steady median an iteration's total is made of. Opponent reconstruction
#: is compile-invariant by construction and belongs to no knob, which is
#: exactly the claim each step's held-phase drift measures.
PHASE_MEDIANS = ("steady_opponent_setup_seconds_median", *KNOB_PHASE_MEDIANS.values())
#: How far a summary's summed phase medians may sit from the steady total
#: median drawn from the same iterations, as a fraction of that total. Matched
#: production reports measure 0.07 %; see `_validate_report` for why the gap is
#: not zero and why it must still be bounded.
MAXIMUM_PHASE_BUDGET_DIVERGENCE = 0.01


class ReportDocument(NamedTuple):
    path: Path
    sha256: str
    size_bytes: int
    records: list[dict[str, Any]]


class ValidatedReport(NamedTuple):
    configuration: dict[str, Any]
    production_summary: dict[str, Any]
    completion: dict[str, Any]


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant in benchmark report: {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key in benchmark report: {key}")
        result[key] = value
    return result


def _read_report(path: Path) -> ReportDocument:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    contents = resolved.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    try:
        lines = contents.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"benchmark report is not UTF-8: {resolved}") from error
    if not lines:
        raise ValueError(f"benchmark report is empty: {resolved}")
    records = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise ValueError(f"benchmark report has a blank record: {resolved}:{line_number}")
        try:
            record = json.loads(
                line,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_object_without_duplicate_keys,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(
                f"invalid benchmark record at {resolved}:{line_number}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"benchmark record {resolved}:{line_number} is not an object")
        records.append(record)
    return ReportDocument(resolved, digest, len(contents), records)


def _validate_json_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"benchmark report has a non-finite value at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"benchmark report has a non-string key at {path}")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise ValueError(f"benchmark report has a non-JSON value at {path}: {type(value).__name__}")


def _configuration(records: list[dict[str, Any]], context: str) -> dict[str, Any]:
    matches = [record for record in records if record.get("event") == "configuration"]
    if len(matches) != 1:
        raise ValueError(
            f"{context} benchmark report must contain exactly one configuration record"
        )
    return matches[0]


def _batch_summary(records: list[dict[str, Any]], games: int) -> dict[str, Any]:
    matches = [
        record
        for record in records
        if record.get("event") == "batch_summary" and record.get("self_play_games") == games
    ]
    if len(matches) != 1:
        raise ValueError(f"benchmark report has no unique {games}-game batch summary")
    return matches[0]


def _require_positive_number(record: dict[str, Any], key: str, context: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{context} {key} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{context} {key} must be finite and positive")
    return converted


def _validate_hardware(hardware: Any, context: str) -> None:
    if not isinstance(hardware, dict):
        raise ValueError(f"{context} hardware identity must be an object")
    for key in ("machine", "processor"):
        if not isinstance(hardware.get(key), str):
            raise ValueError(f"{context} hardware {key} must be a string")
    for key in ("device_name", "torch_cuda_version"):
        if not isinstance(hardware.get(key), str) or not hardware[key]:
            raise ValueError(f"{context} hardware {key} must be a non-empty string")
    if hardware.get("device_type") != "cuda":
        raise ValueError(f"{context} hardware device_type must be 'cuda'")
    for key in ("cpu_count", "total_memory_bytes", "cudnn_version"):
        if type(hardware.get(key)) is not int or hardware[key] <= 0:
            raise ValueError(f"{context} hardware {key} must be a positive integer")
    if type(hardware.get("device_index")) is not int or hardware["device_index"] < 0:
        raise ValueError(f"{context} hardware device_index must be a non-negative integer")
    capability = hardware.get("compute_capability")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or type(capability[0]) is not int
        or type(capability[1]) is not int
        or capability[0] < 1
        or capability[1] < 0
    ):
        raise ValueError(f"{context} hardware compute_capability is invalid")
    if "device_uuid" in hardware and (
        not isinstance(hardware["device_uuid"], str) or not hardware["device_uuid"]
    ):
        raise ValueError(f"{context} hardware device_uuid must be a non-empty string")


def _validate_configuration(
    config: dict[str, Any],
    *,
    knobs: dict[str, Any],
    expected_seed: int,
    context: str,
) -> tuple[list[int], int]:
    expected = {
        "event": "configuration",
        ROLLOUT_FORWARD_MODE_KNOB: knobs[ROLLOUT_FORWARD_MODE_KNOB],
        # Fixed across the chain rather than decided by it. Three reports
        # attribute two knobs because each step moves exactly one, so a third
        # knob would need a fourth report; folding the precision into the mode's
        # value instead would conflate two dimensions and make inductor/fp32
        # unmeasurable. It is a decision measurement already settled outside the
        # chain: the isolated collection forward runs 2.720 ms in fp32 against
        # 1.626 ms in bf16, and matching the update path's precision *lowers*
        # the drift it is audited for -- the shipped parity gate over four
        # production waves measures inductor/bf16 at max_kl 2.2786e-04, tail
        # 0.0 and first_minibatch_kl 4.9855e-03, against eager/fp32's
        # 1.9089e-03, 3.1850e-05 and 4.0598e-02. Faster and 8.4x further inside
        # the gate, so a chain measured in fp32 is not evidence for the run this
        # launches and is rejected here.
        "rollout_bfloat16": PRODUCTION_ROLLOUT_BFLOAT16,
        "device": "cuda",
        "league_games_per_iteration": PRODUCTION_LEAGUE_GAMES,
        "league_opponents": (
            PRODUCTION_LEAGUE_ACTIVE_OPPONENTS + PRODUCTION_LEAGUE_HISTORICAL_OPPONENTS
        ),
        "league_active_opponents": PRODUCTION_LEAGUE_ACTIVE_OPPONENTS,
        "league_historical_opponents": PRODUCTION_LEAGUE_HISTORICAL_OPPONENTS,
        "episode_steps": PRODUCTION_EPISODE_STEPS,
        "seed": expected_seed,
        "temperature": PRODUCTION_TEMPERATURE,
        "opponent_temperature": PRODUCTION_OPPONENT_TEMPERATURE,
        "precision": {
            "use_bfloat16": True,
            "float32_matmul_precision": "high",
            "cudnn_benchmark": True,
        },
        "model": production_model_config(),
        "ppo": production_ppo_config(update_compile_mode=knobs[UPDATE_COMPILE_MODE_KNOB]),
        "max_update_replay_kl": MAX_UPDATE_REPLAY_KL,
        "max_update_replay_tail_fraction": MAX_UPDATE_REPLAY_TAIL_FRACTION,
        "max_first_minibatch_kl": MAX_FIRST_MINIBATCH_KL,
        "max_value_target_saturated_fraction": MAX_VALUE_TARGET_SATURATED_FRACTION,
        "torch": str(torch.__version__),
    }
    for key, expected_value in expected.items():
        if config.get(key) != expected_value:
            raise ValueError(
                f"{context} benchmark {key} does not match production: "
                f"{config.get(key)!r} != {expected_value!r}"
            )
    identity = validate_source_identity(config.get("source_identity"))
    if config.get("source_digest") != identity["sha256"]:
        raise ValueError(f"{context} benchmark source_digest does not match source_identity")
    _validate_hardware(config.get("hardware"), context)

    game_counts = config.get("self_play_game_counts")
    if (
        not isinstance(game_counts, list)
        or not game_counts
        or any(type(value) is not int or value < 1 for value in game_counts)
        or len(set(game_counts)) != len(game_counts)
        or PRODUCTION_SELF_PLAY_GAMES not in game_counts
    ):
        raise ValueError(
            f"{context} benchmark self_play_game_counts must be distinct positive integers "
            f"including {PRODUCTION_SELF_PLAY_GAMES}"
        )
    physical_counts = config.get("physical_games_per_iteration")
    expected_physical = [games + PRODUCTION_LEAGUE_GAMES for games in game_counts]
    if physical_counts != expected_physical:
        raise ValueError(f"{context} benchmark physical game counts are inconsistent")
    repeats = config.get("repeats")
    if type(repeats) is not int or repeats < MINIMUM_CALIBRATION_REPEATS:
        raise ValueError(
            f"{context} benchmark repeats must be an integer of at least "
            f"{MINIMUM_CALIBRATION_REPEATS}; the first is a cold start and is dropped, so "
            f"fewer leaves under {MINIMUM_STEADY_SAMPLES} steady iterations to take a median of"
        )
    return game_counts, repeats


def _expected_completion(game_counts: list[int], repeats: int) -> dict[str, Any]:
    return {
        "event": "benchmark_complete",
        "completed": True,
        "self_play_game_counts": game_counts,
        "repeats": repeats,
        "completed_batches": [
            {
                "self_play_games": games,
                "completed_repeats": list(range(repeats)),
            }
            for games in game_counts
        ],
        "iteration_records": len(game_counts) * repeats,
        "batch_summaries": len(game_counts),
    }


def _validate_report(
    records: list[dict[str, Any]],
    *,
    knobs: dict[str, bool],
    expected_seed: int,
    context: str,
) -> ValidatedReport:
    if not records:
        raise ValueError(f"{context} benchmark report is empty")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{context} benchmark record {index} is not an object")
        _validate_json_value(record, f"{context}[{index}]")
    config = _configuration(records, context)
    if records[0] is not config:
        raise ValueError(f"{context} benchmark configuration must be the first record")
    game_counts, repeats = _validate_configuration(
        config,
        knobs=knobs,
        expected_seed=expected_seed,
        context=context,
    )

    expected_record_count = 2 + len(game_counts) * (repeats + 1)
    if len(records) != expected_record_count:
        raise ValueError(
            f"{context} benchmark report is incomplete: expected {expected_record_count} "
            f"records, found {len(records)}"
        )
    cursor = 1
    summaries: dict[int, dict[str, Any]] = {}
    for games in game_counts:
        iterations = []
        for repeat in range(repeats):
            record = records[cursor]
            cursor += 1
            expected_phase = "cold_start" if repeat == 0 else "steady_state"
            for key, expected_value in (
                ("event", "iteration"),
                ("phase", expected_phase),
                ("repeat", repeat),
                ("self_play_games", games),
                ("league_games", PRODUCTION_LEAGUE_GAMES),
                ("physical_games", games + PRODUCTION_LEAGUE_GAMES),
            ):
                if record.get(key) != expected_value:
                    raise ValueError(
                        f"{context} benchmark iteration {games}/{repeat} has invalid {key}"
                    )
            for key in (
                "opponent_setup_seconds",
                "rollout_seconds",
                "update_replay_parity_seconds",
                "update_seconds",
                "total_seconds",
                "iterations_per_hour",
                "physical_games_per_rollout_second",
                "critic_replayed_states_per_second",
            ):
                _require_positive_number(record, key, f"{context} benchmark iteration")
            if type(record.get("actor_updates")) is not int or record["actor_updates"] < 1:
                raise ValueError(f"{context} benchmark iteration has no actor update")
            # Every knob attribution rests on total_seconds being exactly the
            # three phases summed, which is how the benchmark computes it.
            # Check it here rather than trusting it: without this a report
            # whose phases and total disagree passes every other gate, because
            # each summary median is only checked against the median of the
            # same key. A step's attributed speedup swaps one phase inside a
            # budget summed from those medians, so a report whose phases do not
            # add up would have that budget describe no iteration it contains,
            # and the attribution would silently be arithmetic on unrelated
            # numbers. Tolerance is float-addition slack on a sum of three
            # positive perf_counter deltas, not a measurement allowance.
            summed = (
                record["opponent_setup_seconds"]
                + record["rollout_seconds"]
                + record["update_seconds"]
            )
            if abs(summed - record["total_seconds"]) > 1e-6 * max(summed, 1.0):
                raise ValueError(
                    f"{context} benchmark iteration {games}/{repeat} total seconds "
                    f"{record['total_seconds']} is not its rollout and update phases "
                    f"summed ({summed})"
                )
            # Same reasoning one step further: the derived rate has to describe
            # the total it was derived from, or a forged report can state an
            # honest total and an inflated throughput beside it.
            if abs(record["iterations_per_hour"] * record["total_seconds"] - 3600.0) > 1e-3:
                raise ValueError(
                    f"{context} benchmark iteration {games}/{repeat} iterations per hour "
                    f"{record['iterations_per_hour']} does not match its total seconds"
                )
            iterations.append(record)

        summary = records[cursor]
        cursor += 1
        for key, expected_value in (
            ("event", "batch_summary"),
            ("self_play_games", games),
            ("league_games", PRODUCTION_LEAGUE_GAMES),
            ("physical_games", games + PRODUCTION_LEAGUE_GAMES),
        ):
            if summary.get(key) != expected_value:
                raise ValueError(f"{context} benchmark {games}-game summary has invalid {key}")
        expected_metrics = {
            "cold_total_seconds": iterations[0]["total_seconds"],
            "cold_iterations_per_hour": iterations[0]["iterations_per_hour"],
            "cold_physical_games_per_rollout_second": iterations[0][
                "physical_games_per_rollout_second"
            ],
            "steady_total_seconds_median": statistics.median(
                row["total_seconds"] for row in iterations[1:]
            ),
            "steady_opponent_setup_seconds_median": statistics.median(
                row["opponent_setup_seconds"] for row in iterations[1:]
            ),
            "steady_rollout_seconds_median": statistics.median(
                row["rollout_seconds"] for row in iterations[1:]
            ),
            "steady_update_seconds_median": statistics.median(
                row["update_seconds"] for row in iterations[1:]
            ),
            "steady_iterations_per_hour_median": statistics.median(
                row["iterations_per_hour"] for row in iterations[1:]
            ),
            "steady_physical_games_per_rollout_second_median": statistics.median(
                row["physical_games_per_rollout_second"] for row in iterations[1:]
            ),
            "steady_critic_replayed_states_per_second_median": statistics.median(
                row["critic_replayed_states_per_second"] for row in iterations[1:]
            ),
        }
        for key, expected_value in expected_metrics.items():
            actual = _require_positive_number(summary, key, f"{context} benchmark summary")
            if actual != float(expected_value):
                raise ValueError(
                    f"{context} benchmark {games}-game summary {key} does not match iterations"
                )
        # The decision divides sums of these three medians, so a sum has to
        # describe iterations the report actually timed. Every iteration is
        # checked above to satisfy total = setup + rollout + update exactly,
        # but that constrains the median of the sums, not the sum of the
        # medians: when the phases disagree about which iteration was the slow
        # one the two diverge, without bound if they are anti-correlated, and
        # a budget assembled from three different iterations is a fiction no
        # per-iteration check can catch. Rank agreement is why the real gap is
        # small rather than why it is zero, so this is a bound and not an
        # equality.
        budget = sum(float(summary[key]) for key in PHASE_MEDIANS)
        total = float(summary["steady_total_seconds_median"])
        if abs(budget - total) > MAXIMUM_PHASE_BUDGET_DIVERGENCE * total:
            raise ValueError(
                f"{context} benchmark {games}-game summary phase medians sum to {budget} "
                f"against a steady total median of {total}, so they describe no iteration"
            )
        summaries[games] = summary

    completion = records[cursor]
    expected_completion = _expected_completion(game_counts, repeats)
    if completion != expected_completion:
        raise ValueError(f"{context} benchmark terminal completion record is invalid")
    return ValidatedReport(config, summaries[PRODUCTION_SELF_PLAY_GAMES], completion)


def _chain_context(index: int, length: int) -> str:
    """The name a chain node is reported and complained about under."""
    if index == 0:
        return "eager"
    if index == length - 1:
        return "compiled"
    return "mixed"


def _declared_knobs(records: list[dict[str, Any]], context: str) -> dict[str, Any]:
    """The compilation each report says it ran under, taken from the report.

    The chain's shape is evidence, not an argument: which knob a step turns on
    is read out of the two reports it spans. Passing the shape in beside the
    reports would let a mislabelled pair name a knob neither run moved.

    This is also where both knobs' value domains are enforced, because this is
    the last point in the pipeline that can: `provenance` re-derives the decision
    inside the submission bundle, which ships neither `rollout.py` nor `ppo.py`,
    so it can only check that each knob is a mode-shaped value. A report naming a
    mode outside `ROLLOUT_FORWARD_MODES` or `UPDATE_COMPILE_MODES` therefore has
    to die here, before it becomes a decision.
    """
    config = _configuration(records, context)
    ppo = config.get("ppo")
    declared = {
        ROLLOUT_FORWARD_MODE_KNOB: config.get(ROLLOUT_FORWARD_MODE_KNOB),
        # A schema asymmetry the report producers own: the collection mode sits
        # at top level because it is not a `PpoConfig` field, and the update mode
        # is nested because it is one.
        UPDATE_COMPILE_MODE_KNOB: (
            ppo.get(UPDATE_COMPILE_MODE_KNOB) if isinstance(ppo, dict) else None
        ),
    }
    for knob, domain, description in (
        (ROLLOUT_FORWARD_MODE_KNOB, ROLLOUT_FORWARD_MODES, "a collection forward mode"),
        (UPDATE_COMPILE_MODE_KNOB, UPDATE_COMPILE_MODES, "an update compile mode"),
    ):
        if declared[knob] not in domain:
            raise ValueError(
                f"{context} benchmark configuration does not declare {description} "
                f"from {list(domain)}: {declared[knob]!r}"
            )
    return declared


def _comparable_configuration(config: dict[str, Any]) -> dict[str, Any]:
    """A configuration with only the knobs removed, so everything else must match.

    Only the knobs: the collection precision stays in the comparison, because it
    is not one. It is held at `PRODUCTION_ROLLOUT_BFLOAT16` on every node and
    pinned there by `_validate_configuration`, so a chain that moved it would be
    attributing a precision change to whichever knob its step happened to name.

    The batch-size sweep stays in the comparison. Only the production point's
    median is read below, so excluding it would be sound -- but the cost it was
    excluded to save is better saved by running every node of the chain at the
    production batch, which is what the recipe prescribes. That keeps the
    timings the same protocol as well as the same configuration, which an
    asymmetric sweep cannot claim: the archive pairs single-batch runs at
    2.374-2.418x against full-sweep runs at 2.545-2.694x, and since every
    source digest was measured under exactly one sweep, nothing in it separates
    the protocol from the tree.
    """
    comparable = dict(config)
    del comparable[ROLLOUT_FORWARD_MODE_KNOB]
    comparable["ppo"] = {
        key: value for key, value in comparable["ppo"].items() if key != UPDATE_COMPILE_MODE_KNOB
    }
    return comparable


def choose_compilation(
    chain_records: Sequence[list[dict[str, Any]]],
    *,
    expected_seed: int = 20260812,
) -> dict[str, Any]:
    """Decide each compile knob against the benchmark pair that isolates it.

    The reports form a chain from all-eager to all-compiled in which every step
    turns on exactly one knob and changes nothing else. Two knobs therefore
    need three reports, and the middle one is what makes the decision
    attributable: differencing all-eager against all-compiled moves both knobs
    at once, so each phase's ratio carries whatever between-run drift the pair
    happened to have. That is not hypothetical. The conv calibration this
    design replaces measured the rollout phase at 8.5054 s eager and 8.0847 s
    in a run that also left the rollout uncompiled -- 4.9% apart with the knob
    unchanged -- while crediting the knob itself with 8.0%. Against the report
    that isolates it the rollout knob measures 1.027 and loses; against the
    contaminated pair it measured 1.080 and won.

    Each knob is scored by what it does to the whole iteration, not to its own
    phase: the isolating pair's iteration budget with that knob's phase, and
    only that phase, moved to its measured value under the knob. A per-phase
    ratio flatters a knob whose phase is a small share of the iteration -- the
    rollout knob's 1.027 on an 8 s phase is 1.006 on a 37 s iteration, worth
    under two minutes across a 500-iteration run -- and compilation is not
    free: it costs warmup, replay divergence against the collector, and a
    decision that is stamped into provenance and cannot be revised mid-run.
    Holding every other phase at its measured value on the same side of the
    pair is what keeps the drift out of the number.

    Because each knob is enabled only on top of the knobs before it in the
    chain, the decided configuration is a node of the chain and so was measured
    rather than projected.

    Both knobs are execution modes rather than booleans, so a step turns one from
    `eager` to whichever mode the next node declares, and the decision carries
    that mode rather than a `True`. On the collection side the boolean it
    replaces could only ever mean `cudagraphs`, which measures 5.309 ms against
    eager's 4.907 ms on the isolated forward: the only rollout knob a chain could
    offer was the one mode worth refusing, and no attribution over it could reach
    the 1.626 ms configuration that inductor with bf16 measures. On the update
    side the boolean named four modes at once, so the chain could attribute a
    speedup without recording which of them earned it.
    """
    if type(expected_seed) is not int or expected_seed < 0:
        raise ValueError("expected seed must be a non-negative integer")
    reports = list(chain_records)
    if len(reports) != len(CALIBRATION_KNOBS) + 1:
        raise ValueError(
            f"a calibration chain over {len(CALIBRATION_KNOBS)} knobs needs "
            f"{len(CALIBRATION_KNOBS) + 1} reports, one per configuration from all-eager to "
            f"all-compiled; {len(reports)} were supplied"
        )
    contexts = [_chain_context(index, len(reports)) for index in range(len(reports))]
    declared = [
        _declared_knobs(records, context)
        for records, context in zip(reports, contexts, strict=True)
    ]
    validated = [
        _validate_report(
            records,
            knobs=knobs,
            expected_seed=expected_seed,
            context=context,
        )
        for records, knobs, context in zip(reports, declared, contexts, strict=True)
    ]
    baseline = _comparable_configuration(validated[0].configuration)
    for report, context in zip(validated[1:], contexts[1:], strict=True):
        comparable = _comparable_configuration(report.configuration)
        if comparable != baseline:
            differing = sorted(
                key
                for key in comparable.keys() | baseline.keys()
                if comparable.get(key) != baseline.get(key)
            )
            raise ValueError(
                f"{contexts[0]} and {context} benchmark configurations differ at {differing}"
            )

    enabled = sorted(
        f"{knob}={declared[0][knob]!r}"
        for knob in CALIBRATION_KNOBS
        if declared[0][knob] != UNCOMPILED_CALIBRATION_MODES[knob]
    )
    if enabled:
        raise ValueError(
            f"the calibration chain must start all-eager; its first report has {enabled}"
        )
    steps: list[str] = []
    for index in range(len(CALIBRATION_KNOBS)):
        before, after = declared[index], declared[index + 1]
        moved = [knob for knob in CALIBRATION_KNOBS if after[knob] != before[knob]]
        # A step turns exactly one knob on: that knob leaves its uncompiled
        # value and no knob returns to one. Both knobs are mode-valued, so "on"
        # is not a truth value and both ends are compared against the uncompiled
        # mode rather than tested for truth. Otherwise a step from `cudagraphs`
        # to `inductor`, or from `default` to `max-autotune`, would read as
        # turning its knob on, and the pair said to isolate it would be two
        # compiled runs -- the attribution below substitutes one node's phase
        # median into the other's iteration total, which only answers "this knob
        # or nothing" when one side is nothing.
        if len(moved) != 1 or (
            before[moved[0]] != UNCOMPILED_CALIBRATION_MODES[moved[0]]
            or after[moved[0]] == UNCOMPILED_CALIBRATION_MODES[moved[0]]
        ):
            raise ValueError(
                f"the {contexts[index]} to {contexts[index + 1]} step of the calibration chain "
                f"must turn exactly one knob on; it changes {sorted(moved)}"
            )
        steps.append(moved[0])

    summaries = [report.production_summary for report in validated]
    totals = [float(summary["steady_total_seconds_median"]) for summary in summaries]
    evidence: dict[str, dict[str, Any]] = {}
    for index, knob in enumerate(steps):
        before, after = summaries[index], summaries[index + 1]
        phase = KNOB_PHASE_MEDIANS[knob]
        # The numerator is a measured iteration total, not a sum of phase
        # medians, and the denominator is that same measured total with only
        # this knob's phase replaced by what the step's other node measured for
        # it. So the ratio is the whole-iteration counterfactual "this knob and
        # nothing else", anchored at both ends on an iteration the benchmark
        # actually ran. Substituting one median into another is only meaningful
        # while the phase medians describe the iterations the total was taken
        # from, which `_validate_report` is what enforces.
        attributed = totals[index] / (totals[index] - float(before[phase]) + float(after[phase]))
        evidence[knob] = {
            "isolated_by": [contexts[index], contexts[index + 1]],
            "attributed_iteration_speedup": attributed,
            "phase_speedup": float(before[phase]) / float(after[phase]),
            "step_total_speedup": totals[index] / totals[index + 1],
            # What the pair did to the phases the step did not compile. These
            # should be 1.0 and the amount they are not is this pair's drift,
            # which is the only reason the attributed speedup above is worth
            # more than the step's raw total ratio beside it.
            "held_phase_drift": {
                key: float(after[key]) / float(before[key]) for key in PHASE_MEDIANS if key != phase
            },
        }

    clears = [
        evidence[knob]["attributed_iteration_speedup"] >= MINIMUM_COMPILE_SPEEDUP for knob in steps
    ]
    enabled_steps = clears.index(False) if False in clears else len(clears)
    # A knob that clears the floor behind a knob that does not was measured on
    # top of a configuration this decision will not run, so the chain does not
    # say what it is worth on its own. Refuse rather than guess: the chain that
    # answers it is the one that enables this knob first, and it is one
    # benchmark away.
    late = [
        knob
        for knob, cleared in zip(
            steps[enabled_steps + 1 :], clears[enabled_steps + 1 :], strict=True
        )
        if cleared
    ]
    if late:
        raise ValueError(
            f"the calibration chain enables {sorted(late)} only on top of "
            f"{steps[enabled_steps]}, which does not clear "
            f"{MINIMUM_COMPILE_SPEEDUP}; re-run the chain enabling {sorted(late)} first so each "
            "knob is measured against the configuration that would actually run"
        )
    decided_index = enabled_steps
    # Read back from the node that will run rather than reconstructed: both knobs
    # are mode-valued, so "on" is a particular mode, and the only honest source
    # for which one is the report whose timings decided it.
    decided = dict(declared[decided_index])
    return {
        **decided,
        # Not a knob (see `_validate_configuration`), carried so the launched
        # command states the collection precision its evidence was measured at
        # instead of inheriting whatever the default is that day.
        "rollout_bfloat16": validated[decided_index].configuration["rollout_bfloat16"],
        "minimum_compile_speedup": MINIMUM_COMPILE_SPEEDUP,
        "measured_compile_speedup": totals[0] / totals[-1],
        "attributed_knob_speedups": {
            knob: evidence[knob]["attributed_iteration_speedup"] for knob in CALIBRATION_KNOBS
        },
        "knob_evidence": {knob: evidence[knob] for knob in CALIBRATION_KNOBS},
        "chain": contexts,
        "chain_steps": steps,
        "decided_configuration": contexts[decided_index],
        "decided_steady_total_seconds": totals[decided_index],
        "eager_steady_total_seconds": totals[0],
        "compiled_steady_total_seconds": totals[-1],
        "self_play_games": PRODUCTION_SELF_PLAY_GAMES,
        "league_games": PRODUCTION_LEAGUE_GAMES,
        "validated_evidence": dict(zip(contexts, reports, strict=True)),
    }


def _warm_start_record(
    initial_actor: Path | None, critic_warmup_iterations: int | None
) -> dict[str, Any]:
    return {
        "initial_actor": None if initial_actor is None else str(initial_actor),
        "critic_warmup_iterations": critic_warmup_iterations,
    }


def _recorded_warm_start(decision_path: Path) -> dict[str, Any] | None:
    """The warm start the run being resumed was launched with, if any.

    The decision file is rewritten on every relaunch, so a resume that simply
    restated the current (empty) flags would erase the only launch-side record
    of which clone the weights came from. A decision written before warm
    starting existed carries neither key and is treated as no warm start.
    """
    if not decision_path.is_file():
        return None
    recorded = json.loads(decision_path.read_text(encoding="utf-8"))
    if not isinstance(recorded, dict) or recorded.get("initial_actor") is None:
        return None
    return _warm_start_record(
        Path(str(recorded["initial_actor"])), recorded.get("critic_warmup_iterations")
    )


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _retain_report(document: ReportDocument, destination: Path) -> None:
    """Copy exact benchmark bytes into the run, rejecting provenance collisions."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        contents = destination.read_bytes()
        if (
            hashlib.sha256(contents).hexdigest() != document.sha256
            or len(contents) != document.size_bytes
        ):
            raise FileExistsError(
                f"retained benchmark report conflicts with evidence: {destination}"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(document.path, temporary)
        contents = temporary.read_bytes()
        if (
            hashlib.sha256(contents).hexdigest() != document.sha256
            or len(contents) != document.size_bytes
        ):
            raise ValueError("benchmark report changed while retaining calibration evidence")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eager-report",
        type=Path,
        required=True,
        help="benchmark report with both compile knobs off; the chain's baseline",
    )
    parser.add_argument(
        "--mixed-report",
        type=Path,
        required=True,
        help=(
            "benchmark report with exactly one compile knob on; it is what makes each knob "
            "attributable, since the all-eager and all-compiled reports differ in both at once"
        ),
    )
    parser.add_argument(
        "--compiled-report",
        type=Path,
        required=True,
        help="benchmark report with both compile knobs on",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--max-hours", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--init-actor-from",
        type=Path,
        help=(
            "behavior-cloned actor artifact to initialize iteration zero from; "
            "the critic and both optimizers still start fresh"
        ),
    )
    parser.add_argument(
        "--critic-warmup-iterations",
        type=int,
        help="iterations spent fitting the critic to the warm-started policy before it is trusted",
    )
    return parser.parse_args()


def main() -> None:
    require_repository_launcher(Path(__file__))
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    if not math.isfinite(args.max_hours) or args.max_hours < 0.0:
        raise ValueError("max hours must be finite and non-negative")
    if args.seed < 0:
        raise ValueError("seed cannot be negative")
    requested_actor = (
        None if args.init_actor_from is None else args.init_actor_from.expanduser().resolve()
    )
    if requested_actor is not None and not requested_actor.is_file():
        raise FileNotFoundError(requested_actor)
    if args.critic_warmup_iterations is not None and args.critic_warmup_iterations < 1:
        raise ValueError("critic warmup iterations must be positive")
    if args.critic_warmup_iterations is not None and args.init_actor_from is None:
        raise ValueError("critic warmup applies only to a warm-started run")
    # A warmup that outlives the run never lets the actor take a step, and the
    # stalled-actor guard inside training is suppressed for exactly those
    # iterations, so the run would finish silently identical to its clone.
    if (
        args.critic_warmup_iterations is not None
        and args.critic_warmup_iterations >= args.iterations
    ):
        raise ValueError("critic warmup must leave iterations for the actor to train in")
    documents = {
        "eager": _read_report(args.eager_report),
        "mixed": _read_report(args.mixed_report),
        "compiled": _read_report(args.compiled_report),
    }
    decision = choose_compilation(
        [document.records for document in documents.values()],
        expected_seed=args.seed,
    )
    identity = source_identity()
    benchmark_identity = decision["validated_evidence"]["eager"][0]["source_identity"]
    if identity != benchmark_identity:
        raise ValueError(
            "calibration reports do not match the source tree used to launch training: "
            f"{benchmark_identity['sha256']} != {identity['sha256']}"
        )
    run_directory = args.run_dir.expanduser().resolve()
    resume_checkpoint = resolve_resume_checkpoint(run_directory)
    decision_path = run_directory / "calibration-decision.json"
    # Relaunching the identical command is how a killed run continues, so the
    # warm-start flags must not turn that into an error. Once the run exists
    # its actor and its remaining critic warmup both come from the checkpoint,
    # which is why train_ppo rejects restating them; the launch record still
    # has to name the clone the weights came from, so it is carried forward
    # from the decision this run was started with rather than dropped.
    warm_start = _warm_start_record(requested_actor, args.critic_warmup_iterations)
    initial_actor = None if resume_checkpoint is not None else requested_actor
    critic_warmup_iterations = (
        None if resume_checkpoint is not None else args.critic_warmup_iterations
    )
    if resume_checkpoint is not None:
        warm_start = _recorded_warm_start(decision_path) or warm_start
    evidence_directory = run_directory / "provenance"
    retained = {name: evidence_directory / f"{name}-ppo.jsonl" for name in documents}
    for name, document in documents.items():
        _retain_report(document, retained[name])
    command = build_training_command(
        run_directory,
        iterations=args.iterations,
        max_hours=args.max_hours,
        seed=args.seed,
        rollout_forward_mode=decision[ROLLOUT_FORWARD_MODE_KNOB],
        update_compile_mode=decision[UPDATE_COMPILE_MODE_KNOB],
        rollout_bfloat16=decision["rollout_bfloat16"],
        expected_source_digest=identity["sha256"],
        calibration_decision=decision_path,
        resume_checkpoint=resume_checkpoint,
        initial_actor=initial_actor,
        critic_warmup_iterations=critic_warmup_iterations,
    )
    decision.update(
        {
            **{
                key: value
                for name, document in documents.items()
                for key, value in (
                    (f"{name}_report", str(retained[name])),
                    (f"{name}_report_sha256", document.sha256),
                    (f"{name}_report_size_bytes", document.size_bytes),
                )
            },
            "iterations": args.iterations,
            "max_hours": args.max_hours,
            "seed": args.seed,
            "training_command": command,
            **warm_start,
            "resume_checkpoint": None if resume_checkpoint is None else str(resume_checkpoint),
            "source_identity": identity,
        }
    )
    _write_atomic(decision_path, decision)
    print(json.dumps({"event": "calibrated_launch", **decision}, sort_keys=True), flush=True)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
