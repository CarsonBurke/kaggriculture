from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from torch.utils.tensorboard import SummaryWriter


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tensorboard_reader_reports_complete_scalar_ranges(tmp_path: Path) -> None:
    module = _load_script("read_tensorboard")
    logdir = tmp_path / "tensorboard"
    writer = SummaryWriter(logdir)
    writer.add_scalar("loss/holdout_nll", 0.4, 0)
    writer.add_scalar("loss/holdout_nll", 0.2, 1)
    writer.add_scalar("schedule/learning_rate", 0.01, 1)
    writer.close()

    report = module.read_scalars(logdir, {"loss/holdout_nll"})

    assert report["runs"] == [
        {
            "run": ".",
            "scalars": {
                "loss/holdout_nll": {
                    "count": 2,
                    "first": {"step": 0, "value": pytest.approx(0.4)},
                    "latest": {"step": 1, "value": pytest.approx(0.2)},
                    "min": pytest.approx(0.2),
                    "max": pytest.approx(0.4),
                }
            },
        }
    ]
    with pytest.raises(ValueError, match="absent"):
        module.read_scalars(logdir, {"missing/tag"})


def _evaluation(rewards: list[float], *, artifact: str = "a" * 64) -> dict[str, object]:
    games = []
    for index, reward in enumerate(rewards):
        games.append(
            {
                "seed": 10 + index // 2,
                "seat": index % 2,
                "candidate_seat": index % 2,
                "reward": reward,
            }
        )
    return {
        "artifact_provenance": {"sha256": artifact},
        "opponent_label": "public-v27",
        "valid_for_selection": True,
        "games": games,
        "summary": {
            "games": len(games),
            "invalid_games": 0,
            "score_rate": 0.5,
            "mean_margin": 12.0,
            "mean_reward": sum(rewards) / len(rewards),
        },
    }


def test_bc_score_reader_validates_provenance_and_pairs_by_seed(tmp_path: Path) -> None:
    module = _load_script("score_bc_runs")
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    (baseline_dir / "eval-public-v27.json").write_text(
        json.dumps(_evaluation([10.0, 20.0, 30.0, 40.0])), encoding="utf-8"
    )
    (candidate_dir / "eval-public-v27.json").write_text(
        json.dumps(_evaluation([12.0, 22.0, 36.0, 46.0])), encoding="utf-8"
    )

    baseline = module.load_run(baseline_dir)
    candidate = module.load_run(candidate_dir)
    summary = module.summarize_run(
        candidate,
        baseline,
        bootstrap_samples=1_000,
        bootstrap_seed=7,
    )

    paired = summary["opponents"]["public-v27"]["paired_reward_delta"]
    assert paired["mean"] == pytest.approx(4.0)
    assert paired["seed_clusters"] == 2
    assert paired["bootstrap_95ci"] == pytest.approx([2.0, 6.0])
    assert summary["artifact_sha256"] == "a" * 64


def test_autocull_requires_warmup_and_plateau_below_target() -> None:
    module = _load_script("autocull_hook")
    improving = [{"holdout_nll": value} for value in (1.0, 0.8, 0.6, 0.4)]
    plateau = improving + [{"holdout_nll": value} for value in (0.41, 0.405, 0.41)]
    common = {
        "metric": "holdout_nll",
        "mode": "min",
        "warmup": 4,
        "patience": 3,
        "min_improvement": 0.05,
        "ema_alpha": 1.0,
        "target": 0.3,
    }

    assert module.decide(improving[:3], **common)["reason"] == "warmup"
    decision = module.decide(plateau, **common)
    assert decision["decision"] == "cull"
    assert decision["reason"] == "plateau_below_target"
    assert decision["stale_observations"] == 3
    adequate = module.decide([*plateau, {"holdout_nll": 0.29}], **common)
    assert adequate["decision"] == "continue"
    assert adequate["reason"] == "target_reached"


def test_autocull_rejects_nonfinite_metrics() -> None:
    module = _load_script("autocull_hook")
    with pytest.raises(ValueError, match="not finite"):
        module.decide(
            [{"holdout_nll": float("nan")}],
            metric="holdout_nll",
            mode="min",
            warmup=1,
            patience=1,
            min_improvement=0.1,
            ema_alpha=0.3,
            target=None,
        )


def test_autocull_cli_persists_evidence_and_exits_75_on_cull(tmp_path: Path) -> None:
    journal = tmp_path / "metrics.jsonl"
    journal.write_text(
        "".join(json.dumps({"holdout_nll": value}) + "\n" for value in (1.0, 0.5, 0.51, 0.5, 0.52)),
        encoding="utf-8",
    )
    state = tmp_path / "autocull.json"
    script = Path(__file__).parents[1] / "scripts" / "autocull_hook.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(journal),
            "--warmup",
            "3",
            "--patience",
            "3",
            "--min-improvement",
            "0.1",
            "--ema-alpha",
            "1",
            "--target",
            "0.4",
            "--state",
            str(state),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 75
    payload = json.loads(state.read_text(encoding="utf-8"))
    assert payload["decision"] == "cull"
    assert payload["reason"] == "plateau_below_target"
    assert json.loads(completed.stdout) == payload
