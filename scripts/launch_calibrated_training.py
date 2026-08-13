#!/usr/bin/env python3
"""Launch production VAPO with compilation selected by matched benchmark evidence."""

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
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple

import torch

from kaggriculture.model import ModelConfig
from kaggriculture.provenance import source_identity, validate_source_identity
from kaggriculture.vapo import VapoConfig

PRODUCTION_SELF_PLAY_GAMES = 112
PRODUCTION_LEAGUE_GAMES = 96
PRODUCTION_LEAGUE_INITIAL_OPPONENTS = 1
PRODUCTION_LEAGUE_ACTIVE_OPPONENTS = 2
PRODUCTION_LEAGUE_HISTORICAL_OPPONENTS = 2
PRODUCTION_LEAGUE_ACTIVE_POOL_SIZE = 16
PRODUCTION_EPISODE_STEPS = 720
PRODUCTION_CHECKPOINT_EVERY = 5
PRODUCTION_TEMPERATURE = 1.0
PRODUCTION_OPPONENT_TEMPERATURE = 0.8
MINIMUM_COMPILE_SPEEDUP = 1.05


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


def _configuration(records: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [record for record in records if record.get("event") == "configuration"]
    if len(matches) != 1:
        raise ValueError("benchmark report must contain exactly one configuration record")
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


def _production_model_config() -> dict[str, int | float]:
    return ModelConfig().to_dict()


def _production_vapo_config() -> dict[str, int | float | bool]:
    return asdict(VapoConfig(epochs=1, minibatch_size=2048, target_kl=0.03))


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
    compiled: bool,
    expected_seed: int,
    context: str,
) -> tuple[list[int], int]:
    expected = {
        "event": "configuration",
        "compile_models": compiled,
        "device": "cuda",
        "league_games_per_iteration": PRODUCTION_LEAGUE_GAMES,
        "league_opponents": (
            PRODUCTION_LEAGUE_INITIAL_OPPONENTS
            + PRODUCTION_LEAGUE_ACTIVE_OPPONENTS
            + PRODUCTION_LEAGUE_HISTORICAL_OPPONENTS
        ),
        "league_initial_opponents": PRODUCTION_LEAGUE_INITIAL_OPPONENTS,
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
        "model": _production_model_config(),
        "vapo": _production_vapo_config(),
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
    if type(repeats) is not int or repeats < 2:
        raise ValueError(f"{context} benchmark repeats must be an integer of at least two")
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
    compiled: bool,
    expected_seed: int,
    context: str,
) -> ValidatedReport:
    if not records:
        raise ValueError(f"{context} benchmark report is empty")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{context} benchmark record {index} is not an object")
        _validate_json_value(record, f"{context}[{index}]")
    config = _configuration(records)
    if records[0] is not config:
        raise ValueError(f"{context} benchmark configuration must be the first record")
    game_counts, repeats = _validate_configuration(
        config,
        compiled=compiled,
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
                "self_play_rollout_seconds",
                "league_rollout_seconds",
                "rollout_seconds",
                "update_seconds",
                "total_seconds",
                "iterations_per_hour",
                "physical_games_per_rollout_second",
                "critic_replayed_states_per_second",
            ):
                _require_positive_number(record, key, f"{context} benchmark iteration")
            if type(record.get("actor_updates")) is not int or record["actor_updates"] < 1:
                raise ValueError(f"{context} benchmark iteration has no actor update")
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
        summaries[games] = summary

    completion = records[cursor]
    expected_completion = _expected_completion(game_counts, repeats)
    if completion != expected_completion:
        raise ValueError(f"{context} benchmark terminal completion record is invalid")
    return ValidatedReport(config, summaries[PRODUCTION_SELF_PLAY_GAMES], completion)


def choose_compilation(
    eager_records: list[dict[str, Any]],
    compiled_records: list[dict[str, Any]],
    *,
    expected_seed: int = 20260812,
) -> dict[str, Any]:
    """Require matched production evidence and return a deterministic launch decision."""
    if type(expected_seed) is not int or expected_seed < 0:
        raise ValueError("expected seed must be a non-negative integer")
    eager_validated = _validate_report(
        eager_records,
        compiled=False,
        expected_seed=expected_seed,
        context="eager",
    )
    compiled_validated = _validate_report(
        compiled_records,
        compiled=True,
        expected_seed=expected_seed,
        context="compiled",
    )
    eager_comparable = dict(eager_validated.configuration)
    compiled_comparable = dict(compiled_validated.configuration)
    del eager_comparable["compile_models"]
    del compiled_comparable["compile_models"]
    if eager_comparable != compiled_comparable:
        differing = sorted(
            key
            for key in eager_comparable.keys() | compiled_comparable.keys()
            if eager_comparable.get(key) != compiled_comparable.get(key)
        )
        raise ValueError(f"eager and compiled benchmark configurations differ at {differing}")

    eager_seconds = float(eager_validated.production_summary["steady_total_seconds_median"])
    compiled_seconds = float(compiled_validated.production_summary["steady_total_seconds_median"])
    speedup = eager_seconds / compiled_seconds
    enabled = speedup >= MINIMUM_COMPILE_SPEEDUP
    return {
        "compile_models": enabled,
        "minimum_compile_speedup": MINIMUM_COMPILE_SPEEDUP,
        "measured_compile_speedup": speedup,
        "eager_steady_total_seconds": eager_seconds,
        "compiled_steady_total_seconds": compiled_seconds,
        "self_play_games": PRODUCTION_SELF_PLAY_GAMES,
        "league_games": PRODUCTION_LEAGUE_GAMES,
        "validated_evidence": {
            "eager": eager_records,
            "compiled": compiled_records,
        },
    }


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
    parser.add_argument("--eager-report", type=Path, required=True)
    parser.add_argument("--compiled-report", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--max-hours", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def _training_command(
    args: argparse.Namespace,
    run_directory: Path,
    *,
    compile_models: bool,
    expected_source_digest: str,
    calibration_decision: Path,
    resume_checkpoint: Path | None = None,
) -> list[str]:
    model = _production_model_config()
    vapo = _production_vapo_config()
    command = [
        sys.executable,
        str(Path(__file__).with_name("train_vapo.py")),
        "--run-dir",
        str(run_directory),
        "--iterations",
        str(args.iterations),
        "--max-hours",
        str(args.max_hours),
        "--seed",
        str(args.seed),
        "--expected-source-digest",
        expected_source_digest,
        "--calibration-decision",
        str(calibration_decision),
        "--device",
        "cuda",
        "--games",
        str(PRODUCTION_SELF_PLAY_GAMES),
        "--league-games",
        str(PRODUCTION_LEAGUE_GAMES),
        "--league-active-opponents",
        str(PRODUCTION_LEAGUE_ACTIVE_OPPONENTS),
        "--league-historical-opponents",
        str(PRODUCTION_LEAGUE_HISTORICAL_OPPONENTS),
        "--league-active-pool-size",
        str(PRODUCTION_LEAGUE_ACTIVE_POOL_SIZE),
        "--opponent-temperature",
        str(PRODUCTION_OPPONENT_TEMPERATURE),
        "--episode-steps",
        str(PRODUCTION_EPISODE_STEPS),
        "--temperature",
        str(PRODUCTION_TEMPERATURE),
        "--checkpoint-every",
        str(PRODUCTION_CHECKPOINT_EVERY),
        "--cnn-width",
        str(model["cnn_width"]),
        "--cnn-blocks",
        str(model["cnn_blocks"]),
        "--model-dim",
        str(model["model_dim"]),
        "--transformer-layers",
        str(model["transformer_layers"]),
        "--attention-heads",
        str(model["attention_heads"]),
        "--ffn-multiplier",
        str(model["ffn_multiplier"]),
        "--quantity-rank",
        str(model["quantity_rank"]),
        "--actor-lr",
        str(vapo["actor_learning_rate"]),
        "--critic-lr",
        str(vapo["critic_learning_rate"]),
        "--lr-warmup-steps",
        str(vapo["lr_warmup_steps"]),
        "--weight-decay",
        str(vapo["weight_decay"]),
        "--epochs",
        str(vapo["epochs"]),
        "--minibatch-size",
        str(vapo["minibatch_size"]),
        "--clip-low",
        str(vapo["clip_low"]),
        "--clip-high",
        str(vapo["clip_high"]),
        "--gamma",
        str(vapo["gamma"]),
        "--actor-gae-lambda",
        str(vapo["actor_gae_lambda"]),
        "--target-kl",
        str(vapo["target_kl"]),
        "--max-gradient-norm",
        str(vapo["max_gradient_norm"]),
    ]
    if compile_models:
        command.append("--compile-models")
    if resume_checkpoint is not None:
        command.extend(("--resume", str(resume_checkpoint)))
    return command


def main() -> None:
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    if not math.isfinite(args.max_hours) or args.max_hours < 0.0:
        raise ValueError("max hours must be finite and non-negative")
    if args.seed < 0:
        raise ValueError("seed cannot be negative")
    eager_report = _read_report(args.eager_report)
    compiled_report = _read_report(args.compiled_report)
    decision = choose_compilation(
        eager_report.records,
        compiled_report.records,
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
    latest_checkpoint = run_directory / "latest.pt"
    if latest_checkpoint.is_symlink() or (
        latest_checkpoint.exists() and not latest_checkpoint.is_file()
    ):
        raise ValueError(f"training resume checkpoint is not a regular file: {latest_checkpoint}")
    resume_checkpoint = latest_checkpoint if latest_checkpoint.is_file() else None
    evidence_directory = run_directory / "provenance"
    eager_retained = evidence_directory / "eager-vapo.jsonl"
    compiled_retained = evidence_directory / "compiled-vapo.jsonl"
    _retain_report(eager_report, eager_retained)
    _retain_report(compiled_report, compiled_retained)
    decision_path = run_directory / "calibration-decision.json"
    command = _training_command(
        args,
        run_directory,
        compile_models=bool(decision["compile_models"]),
        expected_source_digest=identity["sha256"],
        calibration_decision=decision_path,
        resume_checkpoint=resume_checkpoint,
    )
    decision.update(
        {
            "eager_report": str(eager_retained),
            "eager_report_sha256": eager_report.sha256,
            "eager_report_size_bytes": eager_report.size_bytes,
            "compiled_report": str(compiled_retained),
            "compiled_report_sha256": compiled_report.sha256,
            "compiled_report_size_bytes": compiled_report.size_bytes,
            "iterations": args.iterations,
            "max_hours": args.max_hours,
            "seed": args.seed,
            "training_command": command,
            "resume_checkpoint": None if resume_checkpoint is None else str(resume_checkpoint),
            "source_identity": identity,
        }
    )
    _write_atomic(decision_path, decision)
    print(json.dumps({"event": "calibrated_launch", **decision}, sort_keys=True), flush=True)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
