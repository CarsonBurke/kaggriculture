from __future__ import annotations

import importlib.util
import itertools
import json
import re
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from kaggle_environments import make
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from kaggriculture.inference import load_actor_artifact
from kaggriculture.model import ModelConfig
from kaggriculture.registry import CONV_ENTITY, STRUCTURED
from kaggriculture.structured import StructuredActor, StructuredConfig

EPISODE_STEPS = 8


def _load_trainer():
    path = Path(__file__).parents[1] / "scripts" / "train_bc.py"
    spec = importlib.util.spec_from_file_location("train_bc", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_extractor():
    path = Path(__file__).parents[1] / "scripts" / "extract_bc_dataset.py"
    spec = importlib.util.spec_from_file_location("extract_bc_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory) -> Path:
    """A tiny real-engine mirrored dataset: 2 seeds x 2 seats of starter play."""
    extractor = _load_extractor()
    directory = tmp_path_factory.mktemp("bc-dataset")
    episodes = []
    for seed in (3, 4):
        environment = make(
            "kaggriculture",
            configuration={"episodeSteps": EPISODE_STEPS, "seed": seed},
            debug=False,
        )
        environment.run(["starter", "starter"])
        for seat in (0, 1):
            arrays = extractor.extract_episode(environment.steps, seat, episode_steps=EPISODE_STEPS)
            name = f"episode-{seed:08d}-seat{seat}"
            np.savez_compressed(directory / f"{name}.npz", **arrays)
            episodes.append({"file": f"{name}.npz", "seed": seed, "seat": seat})
    manifest = {
        "format_version": 1,
        "teacher": {"label": "starter", "sha256": None},
        "opponent": {"label": "starter", "sha256": None},
        "episode_steps": EPISODE_STEPS,
        "episodes": episodes,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def _copy_dataset(
    source: Path, destination: Path, seed_shift: int = 0, **manifest_changes: object
) -> Path:
    """A second corpus on disk: the same episode-seats, its own manifest.

    Real mixtures pair one teacher against different opponents, which only the
    manifest records; the episode payloads do not decide how directories merge
    or split, so they are reused rather than replayed. `seed_shift` renumbers
    this corpus's seeds, which is how independent extractions overlap.
    """
    destination.mkdir()
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["episodes"]:
        shutil.copyfile(source / entry["file"], destination / entry["file"])
        entry["seed"] += seed_shift
    manifest.update(manifest_changes)
    (destination / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return destination


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        cnn_width=16, cnn_blocks=1, model_dim=32, transformer_layers=3, attention_heads=4
    )


def _tiny_structured_config() -> StructuredConfig:
    return StructuredConfig(
        model_dim=16,
        attention_heads=2,
        ffn_multiplier=1,
        farm_blocks=1,
        opponent_latents=2,
        latents=4,
        core_layers=1,
        quantity_rank=4,
    )


def test_load_dataset_holds_out_whole_seeds(dataset_dir: Path) -> None:
    trainer = _load_trainer()

    train_split, holdout_split, records = trainer.load_dataset(
        [dataset_dir],
        architecture=CONV_ENTITY,
        holdout_seeds=1,
        encode_workers=1,
    )

    rows_per_seed = 2 * (EPISODE_STEPS - 1)
    assert train_split.rows == rows_per_seed
    assert holdout_split.rows == rows_per_seed
    assert [record["teacher"]["label"] for record in records] == ["starter"]
    assert train_split.staged["board"].dtype == torch.float16
    assert train_split.staged["unit_actions"].dtype == torch.int8


@pytest.mark.parametrize("architecture", [CONV_ENTITY, STRUCTURED])
def test_only_the_minibatch_crosses_to_the_accelerator(
    dataset_dir: Path, architecture: str
) -> None:
    """The corpus stays on the host and the transfer happens per minibatch.

    Staged on the device, dataset size and batch size compete for the same
    memory and a large corpus fails to clone at a batch size that fits by
    itself. The meta device stands in for an accelerator here: it records
    placement without allocating, so the assertion runs on any machine.
    """
    trainer = _load_trainer()
    train_split, _, _ = trainer.load_dataset(
        [dataset_dir],
        architecture=architecture,
        holdout_seeds=1,
        encode_workers=1,
    )
    accelerator = torch.device("meta")

    actor_args, factors = trainer._batch(architecture, train_split, torch.arange(8), accelerator)

    assert all(value.device.type == "cpu" for value in train_split.staged.values())
    assert all(value.device == accelerator for value in factors.values())
    # The conv family splats four tensors; the structured family splats one
    # named tuple of them.
    moved = actor_args if architecture == CONV_ENTITY else tuple(actor_args[0])
    assert moved and all(value.device == accelerator for value in moved)
    assert factors["unit_actions"].dtype == torch.long
    assert moved[0].dtype == (torch.float32 if architecture == CONV_ENTITY else torch.long)


def test_structured_retokenization_yields_matched_rows(dataset_dir: Path) -> None:
    """Both families clone the same episodes: identical row counts and factors."""
    trainer = _load_trainer()

    conv_train, conv_holdout, _ = trainer.load_dataset(
        [dataset_dir],
        architecture=CONV_ENTITY,
        holdout_seeds=1,
        encode_workers=1,
    )
    train_split, holdout_split, _ = trainer.load_dataset(
        [dataset_dir],
        architecture=STRUCTURED,
        holdout_seeds=1,
        encode_workers=1,
    )

    assert (train_split.rows, holdout_split.rows) == (conv_train.rows, conv_holdout.rows)
    for split, conv_split in ((train_split, conv_train), (holdout_split, conv_holdout)):
        for name in trainer._FACTOR_FIELDS:
            torch.testing.assert_close(split.staged[name], conv_split.staged[name])
    assert set(trainer._STRUCTURED_STATE_FIELDS) <= set(train_split.staged)
    assert "board" not in train_split.staged
    assert train_split.staged["tile_categorical"].dtype == torch.int8
    assert train_split.staged["tile_continuous"].dtype == torch.float16


def test_load_dataset_rejects_mask_violating_targets(dataset_dir: Path, tmp_path: Path) -> None:
    trainer = _load_trainer()
    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["episodes"]:
        with np.load(dataset_dir / entry["file"]) as archive:
            arrays = dict(archive)
        arrays["unit_actions"] = arrays["unit_actions"].copy()
        arrays["unit_masks"] = arrays["unit_masks"].copy()
        arrays["unit_masks"][0, 0, arrays["unit_actions"][0, 0]] = False
        np.savez_compressed(corrupt / entry["file"], **arrays)
    (corrupt / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="violate their own masks"):
        trainer.load_dataset(
            [corrupt],
            architecture=CONV_ENTITY,
            holdout_seeds=1,
            encode_workers=1,
        )


def test_load_dataset_merges_directories_and_holds_out_within_each(
    dataset_dir: Path, tmp_path: Path
) -> None:
    """Two corpora become one, and each contributes its own held-out seeds.

    The corpora here carry seeds (3, 4) and (4, 5): extractions choose their
    own `--seed-start` and nothing makes them disjoint. Keyed on the seed
    alone, the second corpus's seed 4 would follow the first's into the
    holdout; keyed globally on the highest pairs, only seed 5 of the second
    corpus would be held out and the holdout would measure one opponent while
    training on the mixture.
    """
    trainer = _load_trainer()
    second = _copy_dataset(
        dataset_dir,
        tmp_path / "vs-pass",
        seed_shift=1,
        opponent={"label": "pass", "sha256": None},
    )

    train_split, holdout_split, records = trainer.load_dataset(
        [dataset_dir, second],
        architecture=CONV_ENTITY,
        holdout_seeds=1,
        encode_workers=1,
    )

    rows_per_seed = 2 * (EPISODE_STEPS - 1)
    assert train_split.rows == 2 * rows_per_seed
    assert holdout_split.rows == 2 * rows_per_seed
    assert sum(record["episodes"] for record in records) == 8
    assert [record["episodes"] for record in records] == [4, 4]
    assert [record["opponent"]["label"] for record in records] == ["starter", "pass"]
    assert [record["path"] for record in records] == [
        str(dataset_dir.resolve()),
        str(second.resolve()),
    ]


def test_load_dataset_rejects_an_empty_dataset_list(dataset_dir: Path) -> None:
    trainer = _load_trainer()

    with pytest.raises(ValueError, match="no dataset directories given"):
        trainer.load_dataset(
            [],
            architecture=CONV_ENTITY,
            holdout_seeds=1,
            encode_workers=1,
        )


def test_load_dataset_rejects_a_directory_of_another_format_version(
    dataset_dir: Path, tmp_path: Path
) -> None:
    trainer = _load_trainer()
    future = _copy_dataset(dataset_dir, tmp_path / "v2", format_version=2)

    with pytest.raises(
        ValueError, match=rf"{re.escape(str(future))}: unsupported dataset format: 2"
    ):
        trainer.load_dataset(
            [dataset_dir, future],
            architecture=CONV_ENTITY,
            holdout_seeds=1,
            encode_workers=1,
        )


def test_load_dataset_rejects_disagreeing_episode_steps(dataset_dir: Path, tmp_path: Path) -> None:
    """Mixing horizons would silently change what one cloned step is."""
    trainer = _load_trainer()
    longer = _copy_dataset(dataset_dir, tmp_path / "longer", episode_steps=EPISODE_STEPS + 1)

    with pytest.raises(
        ValueError, match=rf"episode_steps {EPISODE_STEPS + 1} disagrees with {EPISODE_STEPS}"
    ):
        trainer.load_dataset(
            [dataset_dir, longer],
            architecture=CONV_ENTITY,
            holdout_seeds=1,
            encode_workers=1,
        )


def _tagging_encoder() -> Callable[[str, str], dict[str, np.ndarray]]:
    """A stand-in tokenizer that makes the split assignment observable.

    The fixture's 8-step starter episodes encode byte-identically for every
    seed and seat, so the staged payload cannot say which episode landed on
    which side of the split; one row per episode-seat tagged with load order
    can. The arrays are the minimum `load_dataset` inspects: one active
    component per factor, selected under an all-permitting mask.
    """
    order = itertools.count()

    def encode(path_text: str, architecture_name: str) -> dict[str, np.ndarray]:
        return {
            "tag": np.array([[next(order)]], dtype=np.int64),
            "unit_actions": np.zeros((1, 1), dtype=np.int8),
            "unit_masks": np.ones((1, 1, 1), dtype=bool),
            "market_kinds": np.zeros((1, 1), dtype=np.int8),
            "market_kind_masks": np.ones((1, 1, 1), dtype=bool),
            "market_quantities": np.zeros((1, 1), dtype=np.int8),
            "market_quantity_masks": np.ones((1, 1, 1), dtype=bool),
            "unit_active": np.ones((1, 1), dtype=bool),
            "market_active": np.ones((1, 1), dtype=bool),
            "market_quantity_active": np.ones((1, 1), dtype=bool),
        }

    return encode


def _tags(split: object) -> list[int]:
    return [int(value) for value in split.staged["tag"].flatten().tolist()]


def test_one_directory_splits_exactly_as_it_did_before_mixing(
    dataset_dir: Path, monkeypatch
) -> None:
    """A single dataset must split identically to the pre-mixture rule — the
    highest `holdout_seeds` seeds held out whole, episodes in manifest order —
    or the clones already measured stop being comparable to new ones."""
    trainer = _load_trainer()
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    held_out = set(sorted({int(entry["seed"]) for entry in manifest["episodes"]})[-1:])
    expected: dict[bool, list[int]] = {False: [], True: []}
    for position, entry in enumerate(manifest["episodes"]):
        expected[int(entry["seed"]) in held_out].append(position)
    monkeypatch.setattr(trainer, "_encode_episode_file", _tagging_encoder())

    train_split, holdout_split, _ = trainer.load_dataset(
        [dataset_dir],
        architecture=CONV_ENTITY,
        holdout_seeds=1,
        encode_workers=1,
    )

    assert expected[False] and expected[True]
    assert _tags(train_split) == expected[False]
    assert _tags(holdout_split) == expected[True]


def test_training_improves_and_saves_a_loadable_artifact(dataset_dir: Path, tmp_path: Path) -> None:
    trainer = _load_trainer()
    output = tmp_path / "run"

    best = trainer.train(
        dataset_dirs=[dataset_dir],
        output_dir=output,
        architecture=CONV_ENTITY,
        config=_tiny_config(),
        holdout_seeds=1,
        epochs=2,
        patience=2,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=0.0,
        seed=0,
        device=torch.device("cpu"),
        encode_workers=1,
    )

    assert set(best) >= {"nll", "unit_nll", "unit_accuracy", "kind_accuracy"}
    records = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["epoch"] for record in records] == [0, 1]
    assert all(np.isfinite(record["train_loss"]) for record in records)

    # TensorBoard is written during the run, not by a conversion step someone
    # has to remember afterwards.
    accumulator = EventAccumulator(str(output / "tensorboard")).Reload()
    assert {"loss/train", "loss/holdout_nll", "schedule/learning_rate"} <= set(
        accumulator.Tags()["scalars"]
    )
    assert [event.step for event in accumulator.Scalars("loss/train")] == [0, 1]
    assert [event.value for event in accumulator.Scalars("loss/train")] == pytest.approx(
        [record["train_loss"] for record in records], rel=1e-6
    )
    # Per-head holdout statistics are runs, so the three share one chart.
    heads = EventAccumulator(str(output / "tensorboard" / "heads" / "unit")).Reload()
    assert "holdout/accuracy" in heads.Tags()["scalars"]

    actor, payload = load_actor_artifact(output / "bc-actor.pt")
    assert payload["architecture"] == CONV_ENTITY
    assert payload["bc_provenance"]["teacher"]["label"] == "starter"
    (only,) = payload["bc_provenance"]["datasets"]
    assert len(only["manifest_sha256"]) == 64
    assert (only["path"], only["episodes"]) == (str(dataset_dir.resolve()), 4)
    assert actor.training is False


def test_structured_training_saves_a_loadable_structured_artifact(
    dataset_dir: Path, tmp_path: Path
) -> None:
    trainer = _load_trainer()
    output = tmp_path / "run-structured"

    best = trainer.train(
        dataset_dirs=[dataset_dir],
        output_dir=output,
        architecture=STRUCTURED,
        config=_tiny_structured_config(),
        holdout_seeds=1,
        epochs=2,
        patience=2,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=0.0,
        seed=0,
        device=torch.device("cpu"),
        encode_workers=1,
    )

    assert np.isfinite(best["nll"])
    actor, payload = load_actor_artifact(output / "bc-actor.pt")
    assert isinstance(actor, StructuredActor)
    assert payload["architecture"] == STRUCTURED
    assert payload["model_config"] == _tiny_structured_config().to_dict()
    assert payload["bc_provenance"]["architecture"] == STRUCTURED
    assert actor.training is False


@pytest.mark.parametrize(
    ("architecture", "expected"), [(CONV_ENTITY, ModelConfig()), (STRUCTURED, StructuredConfig())]
)
def test_unflagged_clone_builds_the_family_default_configuration(
    monkeypatch, tmp_path: Path, architecture: str, expected: object
) -> None:
    """A warm start compares model configurations for equality, so an
    unflagged clone and an unflagged training run must agree by construction."""
    trainer = _load_trainer()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_bc.py",
            "--dataset",
            str(tmp_path),
            "--output",
            str(tmp_path / "run"),
            "--architecture",
            architecture,
        ],
    )
    args = trainer.parse_args()

    config = trainer.model_config_from_args(trainer.resolve_architecture(architecture), args)

    assert config == expected


def test_clone_rejects_a_config_from_another_family(dataset_dir: Path, tmp_path: Path) -> None:
    trainer = _load_trainer()

    with pytest.raises(ValueError, match="does not configure the structured architecture"):
        trainer.train(
            dataset_dirs=[dataset_dir],
            output_dir=tmp_path / "mismatch",
            architecture=STRUCTURED,
            config=_tiny_config(),
            holdout_seeds=1,
            epochs=1,
            patience=1,
            batch_size=8,
            learning_rate=1e-3,
            weight_decay=0.0,
            seed=0,
            device=torch.device("cpu"),
            encode_workers=1,
        )


def test_clone_refuses_to_overwrite_an_existing_run(dataset_dir: Path, tmp_path: Path) -> None:
    """A rerun would clobber an artifact that may beat anything it produces."""
    trainer = _load_trainer()
    output = tmp_path / "run"
    arguments = dict(
        dataset_dirs=[dataset_dir],
        output_dir=output,
        architecture=CONV_ENTITY,
        config=_tiny_config(),
        holdout_seeds=1,
        epochs=1,
        patience=1,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=0.0,
        seed=0,
        device=torch.device("cpu"),
        encode_workers=1,
    )

    trainer.train(**arguments)
    with pytest.raises(FileExistsError, match=r"bc-actor\.pt, metrics\.jsonl"):
        trainer.train(**arguments)


def test_epoch_train_loss_is_the_component_weighted_mean(dataset_dir: Path, tmp_path: Path) -> None:
    """The loss is a mean over active components, so the epoch average must
    weight by that count — not by rows, which vary in how many units act."""
    trainer = _load_trainer()
    output = tmp_path / "run"
    trainer.train(
        dataset_dirs=[dataset_dir],
        output_dir=output,
        architecture=CONV_ENTITY,
        config=_tiny_config(),
        holdout_seeds=1,
        epochs=1,
        patience=1,
        batch_size=1_000_000,  # one minibatch: the epoch mean is that loss exactly
        learning_rate=0.0,  # frozen weights, so the recorded loss is reproducible
        weight_decay=0.0,
        seed=0,
        device=torch.device("cpu"),
        encode_workers=1,
    )
    record = json.loads((output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])

    train_split, _, _ = trainer.load_dataset(
        [dataset_dir],
        architecture=CONV_ENTITY,
        holdout_seeds=1,
        encode_workers=1,
    )
    torch.manual_seed(0)
    from kaggriculture.model import FarmActor

    actor = FarmActor(_tiny_config())
    actor_args, factors = trainer._batch(
        CONV_ENTITY, train_split, torch.arange(train_split.rows), torch.device("cpu")
    )
    loss = trainer._clone_loss(actor, actor_args, factors, autocast=False)

    assert record["train_loss"] == pytest.approx(float(loss.detach()), rel=1e-5)


@pytest.mark.parametrize("architecture", [CONV_ENTITY, STRUCTURED])
def test_clone_loss_is_the_masked_mean_component_nll(dataset_dir: Path, architecture: str) -> None:
    trainer = _load_trainer()
    train_split, _, _ = trainer.load_dataset(
        [dataset_dir],
        architecture=architecture,
        holdout_seeds=1,
        encode_workers=1,
    )
    torch.manual_seed(0)
    if architecture == CONV_ENTITY:
        from kaggriculture.model import FarmActor

        actor = FarmActor(_tiny_config())
    else:
        actor = StructuredActor(_tiny_structured_config())
    indices = torch.arange(train_split.rows)
    actor_args, factors = trainer._batch(architecture, train_split, indices, torch.device("cpu"))

    loss = trainer._clone_loss(actor, actor_args, factors, autocast=False)
    metrics = trainer.evaluate(
        architecture, actor, train_split, batch_size=64, device=torch.device("cpu"), autocast=False
    )

    active_counts = {
        "unit": float(factors["unit_active"].sum()),
        "kind": float(factors["market_active"].sum()),
        "quantity": float(factors["market_quantity_active"].sum()),
    }
    expected = sum(metrics[f"{name}_nll"] * count for name, count in active_counts.items()) / sum(
        active_counts.values()
    )
    assert float(loss.detach()) == pytest.approx(expected, rel=1e-5)


def test_run_record_carries_one_provenance_entry_per_dataset(
    dataset_dir: Path, tmp_path: Path
) -> None:
    """Provenance has to name every corpus that shaped the weights, and must
    not summarize a mixture with a scalar that describes one of them."""
    trainer = _load_trainer()
    second = _copy_dataset(
        dataset_dir,
        tmp_path / "v27-vs-pass",
        teacher={"label": "public-v27", "sha256": "a" * 64},
        opponent={"label": "pass", "sha256": None},
    )
    output = tmp_path / "run-mixed"

    trainer.train(
        dataset_dirs=[dataset_dir, second],
        output_dir=output,
        architecture=CONV_ENTITY,
        config=_tiny_config(),
        holdout_seeds=1,
        epochs=1,
        patience=1,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=0.0,
        seed=0,
        device=torch.device("cpu"),
        encode_workers=1,
    )

    _, payload = load_actor_artifact(output / "bc-actor.pt")
    provenance = payload["bc_provenance"]
    records = provenance["datasets"]
    assert [set(record) for record in records] == [
        {"path", "manifest_sha256", "teacher", "opponent", "episodes"}
    ] * 2
    assert [
        (record["path"], record["teacher"]["label"], record["opponent"]["label"])
        for record in records
    ] == [
        (str(dataset_dir.resolve()), "starter", "starter"),
        (str(second.resolve()), "public-v27", "pass"),
    ]
    assert len({record["manifest_sha256"] for record in records}) == 2
    assert all(len(record["manifest_sha256"]) == 64 for record in records)
    # Two teachers and two opponents: either scalar would name one corpus and
    # misdescribe the run as having cloned it alone.
    assert "teacher" not in provenance
    assert "opponent" not in provenance
