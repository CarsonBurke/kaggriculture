from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kaggriculture.evaluation import SCORE_CONFIDENCE, bounded_mean_interval, seed_protocol


def _selection_script(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    path = scripts / "select_checkpoint.py"
    spec = importlib.util.spec_from_file_location("kaggriculture_select_checkpoint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _evaluation(label: str, scores: list[float], margins: list[float], *, valid: bool = True):
    return {
        "valid_for_selection": valid,
        "seed_start": 0,
        "seed_count": len(scores),
        "opponent": label,
        "opponent_label": label,
        "summary": {
            "score_rate": sum(scores) / len(scores),
            "score_rate_95ci": list(bounded_mean_interval(scores)),
            "score_confidence": SCORE_CONFIDENCE,
            "mean_margin": sum(margins) / len(margins),
            "margin_95ci": [-1.0, 1.0],
            "seed_cluster_statistics": [
                {"seed": index, "score_rate": score, "mean_margin": margin}
                for index, (score, margin) in enumerate(zip(scores, margins, strict=True))
            ],
        },
    }


def test_panel_summary_clusters_seeds_across_fixed_opponents(monkeypatch) -> None:
    module = _selection_script(monkeypatch)

    panel = module.summarize_panel(
        [
            _evaluation("strong", [1.0, 0.0, 1.0, 0.0], [10.0, -2.0, 8.0, -4.0]),
            _evaluation("style", [0.5, 0.5, 1.0, 0.0], [2.0, 2.0, 6.0, -2.0]),
        ]
    )

    assert panel["paired_seed_clusters"] == 4
    assert panel["panel_score_rate"] == pytest.approx(0.5)
    assert panel["panel_mean_margin"] == pytest.approx(2.5)
    assert panel["worst_opponent_score_rate"] == pytest.approx(0.5)
    assert len(panel["opponent_summaries"]) == 2


def test_panel_summary_rejects_invalid_or_misaligned_evidence(monkeypatch) -> None:
    module = _selection_script(monkeypatch)
    with pytest.raises(ValueError, match="invalid"):
        module.summarize_panel([_evaluation("broken", [0.5], [0.0], valid=False)])

    first = _evaluation("first", [0.5, 1.0], [0.0, 1.0])
    second = _evaluation("second", [0.5, 1.0], [0.0, 1.0])
    second["summary"]["seed_cluster_statistics"][1]["seed"] = 7
    with pytest.raises(ValueError, match="incomplete"):
        module.summarize_panel([first, second])


def test_ranking_prefers_confident_panel_strength_before_iteration(monkeypatch) -> None:
    module = _selection_script(monkeypatch)
    stronger = {
        "iteration": 10,
        "panel": {
            "panel_score_rate_95ci": [0.61, 0.80],
            "worst_opponent_score_rate": 0.55,
            "panel_margin_95ci": [2.0, 8.0],
        },
    }
    newer_but_weaker = {
        "iteration": 100,
        "panel": {
            "panel_score_rate_95ci": [0.60, 0.90],
            "worst_opponent_score_rate": 0.90,
            "panel_margin_95ci": [100.0, 200.0],
        },
    }

    assert module._ranking_key(stronger) > module._ranking_key(newer_but_weaker)


def test_atomic_promotion_requires_the_evaluated_checkpoint_digest(monkeypatch, tmp_path) -> None:
    module = _selection_script(monkeypatch)
    source = tmp_path / "checkpoint.pt"
    destination = tmp_path / "best.pt"
    source.write_bytes(b"evaluated")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()

    assert module._copy_atomic(source, destination, expected) == expected
    assert destination.read_bytes() == b"evaluated"
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after evaluation"):
        module._copy_atomic(source, destination, expected)
    assert destination.read_bytes() == b"evaluated"


def test_checkpoint_agents_preserves_single_actor_and_enumerates_population(
    monkeypatch, tmp_path
) -> None:
    module = _selection_script(monkeypatch)
    single = tmp_path / "single.pt"
    population = tmp_path / "population.pt"
    one_member_population = tmp_path / "one-member-population.pt"
    torch.save({"actor": {}}, single)
    torch.save({"agents": [{}, {}, {}]}, population)

    torch.save({"agents": [{}]}, one_member_population)
    assert module._checkpoint_agents(single) == (None,)
    assert module._checkpoint_agents(population) == (0, 1, 2)

    assert module._checkpoint_agents(one_member_population) == (0,)


def test_population_selection_ranks_every_checkpoint_member_pair_globally(
    monkeypatch, tmp_path
) -> None:
    module = _selection_script(monkeypatch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoints = [run_dir / f"checkpoint-{iteration:06d}.pt" for iteration in (1, 2)]
    for checkpoint in checkpoints:
        torch.save({"agents": [{}, {}, {}]}, checkpoint)
    output = tmp_path / "selection.json"
    promoted = tmp_path / "best.pt"
    calls = []
    scores = {
        (checkpoints[0], 0): 0.1,
        (checkpoints[0], 1): 0.2,
        (checkpoints[0], 2): 0.3,
        (checkpoints[1], 0): 0.4,
        (checkpoints[1], 1): 0.9,
        (checkpoints[1], 2): 0.5,
    }

    def fake_evaluate(args):
        key = (args.artifact, args.agent)
        calls.append(key)
        evaluation = _evaluation("starter", [scores[key]], [scores[key]])
        evaluation.update(
            {
                "agent": args.agent,
                "artifact_provenance": {
                    "path": str(args.artifact),
                    "sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
                    "iteration": int(args.artifact.stem.removeprefix("checkpoint-")),
                    "source_identity": {"tree": "same"},
                    "run_provenance": None,
                    "agent": args.agent,
                },
                "opponent_provenance": {"kind": "builtin", "name": "starter"},
                "seed_protocol": seed_protocol("screening", args.seed_start, args.seeds, usage=[]),
            }
        )
        return evaluation

    class Writer:
        def __init__(self, _path):
            pass

        def add_scalar(self, *_args):
            pass

        def flush(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(module, "evaluate", fake_evaluate)
    monkeypatch.setattr(module, "SummaryWriter", Writer)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            run_dir=run_dir,
            pattern="checkpoint-*.pt",
            opponents=["starter"],
            seeds=1,
            seed_start=10_000_000,
            workers=1,
            torch_threads=1,
            device="cpu",
            output=output,
            inference_equivalence=None,
            tensorboard_dir=tmp_path / "tensorboard",
            best_output=promoted,
        ),
    )

    module.main()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert calls == [(checkpoint, agent) for checkpoint in checkpoints for agent in range(3)]
    assert len(report["candidates"]) == 6
    assert {(row["checkpoint"], row["agent"]) for row in report["candidates"]} == {
        (str(checkpoint), agent) for checkpoint in checkpoints for agent in range(3)
    }
    assert report["best_checkpoint"] == str(checkpoints[1])
    assert report["best_agent"] == 1
    assert report["candidates"][0]["agent"] == 1
    assert promoted.read_bytes() == checkpoints[1].read_bytes()
