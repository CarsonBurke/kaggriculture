from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from kaggle_environments import make

from kaggriculture.entity import EntityActor, EntityConfig
from kaggriculture.market_set import MarketSetOrder
from kaggriculture.registry import ENTITY_ATTENTION
from kaggriculture.structured import StructuredDecisionBelief


def _trainer():
    path = Path(__file__).resolve().parents[1] / "scripts/train_bc.py"
    spec = importlib.util.spec_from_file_location("train_bc_market_set", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_interface_three_clone_loss_uses_set_values_and_backpropagates() -> None:
    trainer = _trainer()
    actor = EntityActor(EntityConfig(action_interface=3))
    belief = StructuredDecisionBelief(torch.randn(2, 16, 96), torch.randn(2, 21, 96))
    output = actor.decode_belief(belief)
    unit_masks = torch.zeros(2, 16, 68, dtype=torch.bool)
    unit_masks[..., 0] = True
    set_masks = torch.zeros(2, 21, 101, dtype=torch.bool)
    set_masks[..., 0] = True
    set_masks[:, 0, 1:4] = True
    set_active = torch.zeros(2, 21, dtype=torch.bool)
    set_active[:, 0] = True
    factors = {
        "unit_actions": torch.zeros(2, 16, dtype=torch.long),
        "unit_masks": unit_masks,
        "unit_active": torch.ones(2, 16, dtype=torch.bool),
        "market_set_values": torch.tensor([[3] + [0] * 20] * 2),
        "market_set_masks": set_masks,
        "market_set_active": set_active,
    }
    loss = trainer._clone_loss_from_output(actor, output, factors)
    assert torch.isfinite(loss)
    loss.backward()
    assert actor.market_quantity_value.weight.grad is not None
    assert actor.market_quantity_value.weight.grad[101].abs().sum() > 0


def test_paired_engine_replay_builds_version_three_arrays(tmp_path: Path) -> None:
    extractor = _script("extract_bc_dataset.py")
    builder = _script("build_market_set_corpus.py")
    records = []
    for seed in (13, 14):
        environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": seed})
        environment.run(["starter", "starter"])
        inputs: dict[int, Path] = {}
        for seat in (0, 1):
            path = tmp_path / f"input-{seed}-seat{seat}.npz"
            np.savez_compressed(
                path, **extractor.extract_episode(environment.steps, seat, episode_steps=8)
            )
            inputs[seat] = path
        records.extend(builder._process_seed(seed, inputs, tmp_path, MarketSetOrder()))
    assert len(records) == 4
    for record in records:
        with np.load(tmp_path / record["file"], allow_pickle=False) as archive:
            assert archive["market_set_values"].shape == (7, 21)
            assert archive["market_set_masks"].shape == (7, 21, 101)
            assert archive["market_set_active"].shape == (7, 21)
            assert not archive["market_active"].any()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 3,
                "teacher": {"label": "starter"},
                "opponent": {"label": "starter"},
                "episode_steps": 8,
                "market_set": {"sell_order": "fixed", "hire_last": False},
                "episodes": records,
            }
        )
    )
    trainer = _trainer()
    train, holdout, _ = trainer.load_dataset(
        [tmp_path],
        architecture=ENTITY_ATTENTION,
        holdout_seeds=1,
        encode_workers=1,
        action_interface=3,
    )
    assert train.rows == holdout.rows == 14
    assert train.staged["market_set_values"].shape == (14, 21)
    actor = EntityActor(EntityConfig(action_interface=3))
    metrics = trainer.evaluate(
        ENTITY_ATTENTION,
        actor,
        holdout,
        batch_size=14,
        device=torch.device("cpu"),
        autocast=False,
    )
    assert torch.isfinite(torch.tensor(metrics["nll"]))
    assert "set_accuracy" in metrics


@pytest.mark.parametrize("opponent", ["pass", "random"])
def test_nonmirror_replay_or_live_trace_provides_exact_fills(tmp_path: Path, opponent: str) -> None:
    extractor = _script("extract_bc_dataset.py")
    builder = _script("build_market_set_corpus.py")
    environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 19})
    environment.run(["starter", opponent])
    source = tmp_path / "teacher-seat0.npz"
    np.savez_compressed(source, **extractor.extract_episode(environment.steps, 0, episode_steps=8))
    records = builder._process_seed(
        19,
        {0: source},
        tmp_path,
        MarketSetOrder(),
        teacher="starter",
        opponent=opponent,
        fresh_live=opponent == "random",
    )
    assert len(records) == 1
    with np.load(tmp_path / records[0]["file"], allow_pickle=False) as archive:
        assert archive["market_set_values"].shape == (7, 21)


def test_corpus_cli_requires_explicit_lossy_relabel_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = _script("build_market_set_corpus.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_market_set_corpus.py",
            "--source",
            str(tmp_path / "source"),
            "--output",
            str(tmp_path / "output"),
            "--sell-order",
            "fixed",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        builder.main()
