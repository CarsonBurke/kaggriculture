from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from kaggriculture.production import build_training_command


def test_training_command_resumes_the_latest_atomic_checkpoint(tmp_path: Path) -> None:
    latest = tmp_path / "run" / "latest.pt"
    command = build_training_command(
        latest.parent,
        iterations=500,
        max_hours=0.0,
        seed=7,
        compile_models=False,
        expected_source_digest="a" * 64,
        calibration_decision=tmp_path / "decision.json",
        resume_checkpoint=latest,
    )

    assert command[-2:] == ["--resume", str(latest)]


def test_training_command_requires_digest_and_decision_together(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="together"):
        build_training_command(
            tmp_path,
            iterations=1,
            max_hours=0.0,
            seed=7,
            compile_models=True,
            expected_source_digest="a" * 64,
        )


def _script(name: str = "launch_calibrated_training.py"):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"kaggriculture_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hardware() -> dict[str, object]:
    return {
        "device_type": "cuda",
        "machine": "x86_64",
        "processor": "test-cpu",
        "cpu_count": 12,
        "torch_cuda_version": "12.8",
        "cudnn_version": 91002,
        "device_index": 0,
        "device_name": "Test GPU",
        "compute_capability": [9, 0],
        "total_memory_bytes": 24_000_000_000,
        "device_uuid": "GPU-test",
    }


def _records(
    module,
    *,
    compiled: bool,
    seconds: float,
    seed: int = 20260812,
    source_digest: str | None = None,
) -> list[dict[str, object]]:
    game_counts = [64, 112, 128]
    repeats = 2
    configuration: dict[str, object] = {
        "event": "configuration",
        "compile_models": compiled,
        "device": "cuda",
        "hardware": _hardware(),
        "self_play_game_counts": game_counts,
        "league_games_per_iteration": 96,
        "league_opponents": 5,
        "league_initial_opponents": 1,
        "league_active_opponents": 2,
        "league_historical_opponents": 2,
        "episode_steps": 720,
        "physical_games_per_iteration": [160, 208, 224],
        "repeats": repeats,
        "seed": seed,
        "temperature": 1.0,
        "opponent_temperature": 0.8,
        "precision": {
            "use_bfloat16": True,
            "float32_matmul_precision": "high",
            "cudnn_benchmark": True,
        },
        "model": module.production_model_config(),
        "vapo": module.production_vapo_config(compiled=compiled),
        "max_update_replay_error": module.MAX_UPDATE_REPLAY_RATIO_ERROR,
        "max_first_minibatch_kl": module.MAX_FIRST_MINIBATCH_KL,
        "torch": str(module.torch.__version__),
    }
    identity = module.source_identity()
    configuration["source_identity"] = identity
    configuration["source_digest"] = identity["sha256"] if source_digest is None else source_digest
    records: list[dict[str, object]] = [configuration]
    for games in game_counts:
        steady_seconds = seconds if games == 112 else seconds + games / 1000.0
        iterations = []
        for repeat in range(repeats):
            total_seconds = steady_seconds * 1.2 if repeat == 0 else steady_seconds
            rollout_seconds = total_seconds * 0.7
            record: dict[str, object] = {
                "event": "iteration",
                "phase": "cold_start" if repeat == 0 else "steady_state",
                "repeat": repeat,
                "self_play_games": games,
                "league_games": 96,
                "physical_games": games + 96,
                "rollout_seconds": rollout_seconds,
                "update_replay_parity_seconds": total_seconds * 0.05,
                "update_seconds": total_seconds - rollout_seconds,
                "total_seconds": total_seconds,
                "iterations_per_hour": 3600.0 / total_seconds,
                "physical_games_per_rollout_second": (games + 96) / rollout_seconds,
                "critic_replayed_states_per_second": 1000.0 / (total_seconds - rollout_seconds),
                "actor_updates": 1,
            }
            iterations.append(record)
            records.append(record)
        records.append(
            {
                "event": "batch_summary",
                "self_play_games": games,
                "league_games": 96,
                "physical_games": games + 96,
                "cold_total_seconds": iterations[0]["total_seconds"],
                "cold_iterations_per_hour": iterations[0]["iterations_per_hour"],
                "cold_physical_games_per_rollout_second": iterations[0][
                    "physical_games_per_rollout_second"
                ],
                "steady_total_seconds_median": iterations[1]["total_seconds"],
                "steady_iterations_per_hour_median": iterations[1]["iterations_per_hour"],
                "steady_physical_games_per_rollout_second_median": iterations[1][
                    "physical_games_per_rollout_second"
                ],
                "steady_critic_replayed_states_per_second_median": iterations[1][
                    "critic_replayed_states_per_second"
                ],
            }
        )
    records.append(module._expected_completion(game_counts, repeats))
    return records


def _write_report(path: Path, records: list[dict[str, object]]) -> bytes:
    contents = ("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n").encode()
    path.write_bytes(contents)
    return contents


def test_compile_requires_a_material_matched_speedup() -> None:
    module = _script()

    faster = module.choose_compilation(
        _records(module, compiled=False, seconds=10.0),
        _records(module, compiled=True, seconds=9.0),
    )
    marginal = module.choose_compilation(
        _records(module, compiled=False, seconds=10.0),
        _records(module, compiled=True, seconds=9.6),
    )

    assert faster["compile_models"] is True
    assert faster["measured_compile_speedup"] == pytest.approx(10.0 / 9.0)
    assert marginal["compile_models"] is False
    assert faster["validated_evidence"]["eager"][-1]["completed"] is True


def test_compile_decision_rejects_nonproduction_or_mismatched_configuration() -> None:
    module = _script()
    compiled = _records(module, compiled=True, seconds=9.0)
    compiled[0]["model"] = dict(compiled[0]["model"], cnn_width=128)

    with pytest.raises(ValueError, match="model"):
        module.choose_compilation(_records(module, compiled=False, seconds=10.0), compiled)

    eager = _records(module, compiled=False, seconds=10.0)
    compiled = _records(module, compiled=True, seconds=9.0)
    eager[0]["hardware"] = dict(eager[0]["hardware"], device_name="Other GPU")
    with pytest.raises(ValueError, match=r"configurations differ.*hardware"):
        module.choose_compilation(eager, compiled)

    both_nonproduction = _records(module, compiled=False, seconds=10.0)
    both_nonproduction[0]["vapo"] = dict(both_nonproduction[0]["vapo"], epochs=4)
    with pytest.raises(ValueError, match=r"vapo.*production"):
        module.choose_compilation(
            both_nonproduction,
            _records(module, compiled=True, seconds=9.0),
        )


def test_compile_decision_rejects_incomplete_or_fabricated_summary() -> None:
    module = _script()
    eager = _records(module, compiled=False, seconds=10.0)
    compiled = _records(module, compiled=True, seconds=9.0)

    with pytest.raises(ValueError, match="incomplete"):
        module.choose_compilation(eager[:-1], compiled)

    fabricated = _records(module, compiled=True, seconds=9.0)
    production_summary = next(
        record
        for record in fabricated
        if record.get("event") == "batch_summary" and record.get("self_play_games") == 112
    )
    production_summary["steady_total_seconds_median"] = 1.0
    with pytest.raises(ValueError, match="does not match iterations"):
        module.choose_compilation(eager, fabricated)

    nonfinite = _records(module, compiled=True, seconds=9.0)
    nonfinite[1]["total_seconds"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        module.choose_compilation(eager, nonfinite)


def test_source_digest_is_validated_and_must_match() -> None:
    module = _script()
    with pytest.raises(ValueError, match="source_digest"):
        module.choose_compilation(
            _records(module, compiled=False, seconds=10.0, source_digest="invalid"),
            _records(module, compiled=True, seconds=9.0, source_digest="invalid"),
        )

    with pytest.raises(ValueError, match="source_digest"):
        module.choose_compilation(
            _records(module, compiled=False, seconds=10.0),
            _records(module, compiled=True, seconds=9.0, source_digest="b" * 64),
        )


def test_report_reader_hashes_exact_bytes_and_rejects_nonstandard_json(tmp_path: Path) -> None:
    module = _script()
    report = tmp_path / "report.jsonl"
    contents = _write_report(report, _records(module, compiled=False, seconds=10.0))

    document = module._read_report(report)

    assert document.path == report.resolve()
    assert document.sha256 == hashlib.sha256(contents).hexdigest()
    assert document.size_bytes == len(contents)
    report.write_text('{"event":"configuration","value":NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard JSON"):
        module._read_report(report)


def test_main_persists_hashes_full_evidence_and_explicit_training_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script()
    eager_path = tmp_path / "eager.jsonl"
    compiled_path = tmp_path / "compiled.jsonl"
    eager_records = _records(module, compiled=False, seconds=10.0)
    compiled_records = _records(module, compiled=True, seconds=9.0)
    eager_contents = _write_report(eager_path, eager_records)
    compiled_contents = _write_report(compiled_path, compiled_records)
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    latest_checkpoint = run_directory / "latest.pt"
    latest_checkpoint.write_bytes(b"atomic checkpoint")
    invocation: dict[str, object] = {}

    class Executed(Exception):
        pass

    def fake_execv(executable: str, command: list[str]) -> None:
        invocation.update(executable=executable, command=command)
        raise Executed

    monkeypatch.setattr(module.os, "execv", fake_execv)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_calibrated_training.py",
            "--eager-report",
            str(eager_path),
            "--compiled-report",
            str(compiled_path),
            "--run-dir",
            str(run_directory),
            "--iterations",
            "17",
        ],
    )

    with pytest.raises(Executed):
        module.main()

    decision = json.loads((run_directory / "calibration-decision.json").read_text())
    assert decision["eager_report_sha256"] == hashlib.sha256(eager_contents).hexdigest()
    assert decision["compiled_report_sha256"] == hashlib.sha256(compiled_contents).hexdigest()
    assert decision["eager_report_size_bytes"] == len(eager_contents)
    assert decision["compiled_report_size_bytes"] == len(compiled_contents)
    assert decision["validated_evidence"] == {
        "eager": eager_records,
        "compiled": compiled_records,
    }
    assert Path(decision["eager_report"]).read_bytes() == eager_contents
    assert Path(decision["compiled_report"]).read_bytes() == compiled_contents
    assert invocation["executable"] == sys.executable
    assert invocation["command"] == decision["training_command"]
    assert decision["resume_checkpoint"] == str(latest_checkpoint)
    assert decision["training_command"][-2:] == ["--resume", str(latest_checkpoint)]
    assert "--compile-models" in decision["training_command"]
    assert "--entropy-coefficient" not in decision["training_command"]
    assert "entropy_coefficient" not in module.production_vapo_config(compiled=False)
    assert decision["source_identity"] == module.source_identity()
    digest_index = decision["training_command"].index("--expected-source-digest")
    assert decision["training_command"][digest_index + 1] == module.source_identity()["sha256"]
    for flag, expected in (
        ("--games", "112"),
        ("--league-games", "96"),
        ("--league-active-opponents", "2"),
        ("--league-historical-opponents", "2"),
        ("--epochs", "1"),
        ("--minibatch-size", "2048"),
        ("--gamma", "1.0"),
        (
            "--actor-gae-lambda",
            str(module.production_vapo_config(compiled=False)["actor_gae_lambda"]),
        ),
        ("--target-kl", "0.03"),
    ):
        index = decision["training_command"].index(flag)
        assert decision["training_command"][index + 1] == expected


def test_direct_launch_compiles_without_calibration_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script("launch_production.py")
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    latest_checkpoint = run_directory / "latest.pt"
    latest_checkpoint.write_bytes(b"atomic checkpoint")
    invocation: dict[str, object] = {}

    class Executed(Exception):
        pass

    def fake_execv(executable: str, command: list[str]) -> None:
        invocation.update(executable=executable, command=command)
        raise Executed

    monkeypatch.setattr(module.os, "execv", fake_execv)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_production.py",
            "--run-dir",
            str(run_directory),
            "--iterations",
            "17",
        ],
    )

    with pytest.raises(Executed):
        module.main()

    launch = json.loads((run_directory / "launch.json").read_text())
    assert launch["event"] == "direct_launch"
    assert launch["compile_models"] is True
    assert launch["iterations"] == 17
    assert launch["resume_checkpoint"] == str(latest_checkpoint)
    assert launch["source_identity"] == module.source_identity()
    assert invocation["executable"] == sys.executable
    assert invocation["command"] == launch["training_command"]
    assert "--compile-models" in launch["training_command"]
    assert "--expected-source-digest" not in launch["training_command"]
    assert "--calibration-decision" not in launch["training_command"]
    assert launch["training_command"][-2:] == ["--resume", str(latest_checkpoint)]
