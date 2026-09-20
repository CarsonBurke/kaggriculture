from __future__ import annotations

import importlib.util
import inspect
import itertools
import json
import re
import shutil
import sys
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from kaggle_environments import make
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from kaggriculture.inference import load_actor_artifact
from kaggriculture.model import ModelConfig
from kaggriculture.optim import NorMuon
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
        matrix_learning_rate=1e-3,
        # The shipped decay values, so the end-to-end path that must still
        # improve the loss is the cautious one a real clone runs.
        matrix_weight_decay=1.2,
        adam_learning_rate_ratio=0.35,
        adam_weight_decay=0.005,
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
    # Per-head holdout statistics share the run's single event file, with the
    # head in the category so the three draw as adjacent sibling charts. They
    # were separate runs once, which cost the mirror an event file per head.
    assert {"holdout-unit/accuracy", "holdout-kind/accuracy", "holdout-unit/nll"} <= set(
        accumulator.Tags()["scalars"]
    )
    assert not (output / "tensorboard" / "heads").exists()

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
        matrix_learning_rate=1e-3,
        matrix_weight_decay=0.0,
        adam_learning_rate_ratio=0.35,
        adam_weight_decay=0.0,
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


def test_clone_help_formats_literal_percentages(monkeypatch, capsys) -> None:
    trainer = _load_trainer()
    monkeypatch.setattr(sys, "argv", ["train_bc.py", "--help"])

    with pytest.raises(SystemExit) as raised:
        trainer.parse_args()

    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    assert "99.996% accuracy" in help_text
    assert "0.6% of throughput" in help_text


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
            matrix_learning_rate=1e-3,
            matrix_weight_decay=0.0,
            adam_learning_rate_ratio=0.35,
            adam_weight_decay=0.0,
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
        matrix_learning_rate=1e-3,
        matrix_weight_decay=0.0,
        adam_learning_rate_ratio=0.35,
        adam_weight_decay=0.0,
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
        # Any rate: with a single minibatch the recorded loss is the loss at the
        # initial weights, taken before the only step of the epoch.
        matrix_learning_rate=1e-3,
        matrix_weight_decay=0.0,
        adam_learning_rate_ratio=0.35,
        adam_weight_decay=0.0,
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
        matrix_learning_rate=1e-3,
        matrix_weight_decay=0.0,
        adam_learning_rate_ratio=0.35,
        adam_weight_decay=0.0,
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


def test_staging_preserves_row_order_and_releases_each_episode() -> None:
    """Staging is the corpus's memory ceiling, so it must not hold parts and whole.

    A 2,048-seat mixture was killed by the kernel because `np.concatenate` keeps
    every episode alive alongside the copy it builds. Filling a preallocated array
    only helps if the caller's references die with each copy, so the release is
    part of the contract, and so is the row order it must not disturb.
    """
    module = _load_trainer()
    rng = np.random.default_rng(0)
    members = [
        {
            "unit_actions": rng.integers(0, 3, size=(rows, 4), dtype=np.int8),
            "unit_active": rng.integers(0, 2, size=(rows, 4)).astype(bool),
            "market_active": rng.integers(0, 2, size=(rows, 2)).astype(bool),
            "market_quantity_active": rng.integers(0, 2, size=(rows, 2)).astype(bool),
            "board": rng.random((rows, 3, 2, 2)).astype(np.float16),
        }
        for rows in (5, 3, 7)
    ]
    expected = {name: np.concatenate([member[name] for member in members]) for name in members[0]}
    expected_components = sum(
        expected[name].astype(np.float64).sum(axis=1)
        for name in ("unit_active", "market_active", "market_quantity_active")
    )

    staged = module._stage_split(members)

    assert staged.rows == 15
    for name, value in expected.items():
        np.testing.assert_array_equal(staged.staged[name].numpy(), value)
        assert staged.staged[name].dtype == torch.from_numpy(value).dtype
    np.testing.assert_allclose(staged.row_components, expected_components)
    # Every episode released: a member that still holds its arrays is a member the
    # allocator cannot reclaim while the whole-corpus copy is being built.
    assert members == [{}, {}, {}]


def test_a_seed_cap_takes_a_stable_prefix_of_each_corpus(dataset_dir: Path, tmp_path: Path) -> None:
    """The cap must select a prefix, so a directory can grow without moving it.

    The corpus is staged whole in host memory, so a wide mixture needs a cap --
    an uncapped four-opponent mixture was killed by the kernel. Taking the lowest
    seeds means extending a directory later adds episodes rather than silently
    reshuffling which ones a previous run trained on, and the holdout still comes
    off the top of what is kept rather than off seeds the cap discarded.
    """
    trainer = _load_trainer()
    directory = _copy_dataset(dataset_dir, tmp_path / "three-seeds")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    # A third seed, so a cap of two has a seed to discard. Its arrays are the
    # lowest seed's; only the seed the manifest reports decides what the cap keeps.
    lowest = min(int(entry["seed"]) for entry in manifest["episodes"])
    highest = max(int(entry["seed"]) for entry in manifest["episodes"]) + 1
    for seat in (0, 1):
        shutil.copyfile(
            directory / f"episode-{lowest:08d}-seat{seat}.npz",
            directory / f"episode-{highest:08d}-seat{seat}.npz",
        )
        manifest["episodes"].append(
            {"file": f"episode-{highest:08d}-seat{seat}.npz", "seed": highest, "seat": seat}
        )
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    train_split, holdout_split, records = trainer.load_dataset(
        [directory],
        architecture=CONV_ENTITY,
        holdout_seeds=1,
        encode_workers=1,
        seeds_per_dataset=2,
    )

    # Two seeds of two seats kept, the highest seed discarded entirely.
    assert records[0]["episodes"] == 4
    assert records[0]["seeds_kept"] == 2
    assert train_split.rows == 2 * (EPISODE_STEPS - 1)
    assert holdout_split.rows == 2 * (EPISODE_STEPS - 1)
    # The holdout comes off the top of what survived the cap, never off a seed
    # the cap removed -- otherwise a capped run would train on everything it kept.
    with pytest.raises(ValueError, match="holdout of 2 seeds needs"):
        trainer.load_dataset(
            [directory],
            architecture=CONV_ENTITY,
            holdout_seeds=2,
            encode_workers=1,
            seeds_per_dataset=2,
        )


# Uneven episode-seats, including one of a single row: a real corpus's seats are
# equal-length, but the sampler must not depend on that, and the tail arithmetic
# is where a run length that divides nothing goes wrong.
_RUN_SPANS = (7, 5, 1, 12)


def _synthetic_metadata(spans: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Staged `episode_index`/`step` for episode-seats of the given row counts."""
    episode_index = torch.repeat_interleave(
        torch.arange(len(spans), dtype=torch.int32), torch.tensor(spans)
    )
    return episode_index, torch.cat([torch.arange(span, dtype=torch.int32) for span in spans])


@pytest.mark.parametrize("run_length", [1, 2, 3, 7, 64])
def test_an_epoch_under_the_run_sampler_visits_every_row_exactly_once(run_length: int) -> None:
    """An epoch is a pass over the corpus, so the order must be a permutation of
    the rows -- a multiset equality, which a matching row count would not catch,
    and which is why a short tail run is kept whole rather than dropped."""
    trainer = _load_trainer()
    episode_index, _ = _synthetic_metadata(_RUN_SPANS)
    starts, lengths = trainer._run_blocks(episode_index, run_length)

    order = trainer._run_epoch_order(starts, lengths, torch.Generator(device="cpu").manual_seed(0))

    rows = sum(_RUN_SPANS)
    assert sorted(order.tolist()) == list(range(rows))
    assert int(lengths.sum()) == rows


@pytest.mark.parametrize("run_length", [2, 3, 5, 64])
def test_a_run_is_consecutive_steps_of_one_episode_seat(run_length: int) -> None:
    """Pairing reads adjacent rows, so a run must stay inside one episode-seat and
    advance the step by exactly one: a run spanning a boundary would pair the last
    state of one game with the first state of another and call it a transition."""
    trainer = _load_trainer()
    episode_index, step = _synthetic_metadata(_RUN_SPANS)
    starts, lengths = trainer._run_blocks(episode_index, run_length)

    order = trainer._run_epoch_order(starts, lengths, torch.Generator(device="cpu").manual_seed(1))

    for start, length in zip(starts.tolist(), lengths.tolist(), strict=True):
        assert length <= run_length
        rows = torch.arange(start, start + length)
        assert episode_index[rows].unique().numel() == 1
        assert torch.equal(torch.diff(step[rows]), torch.ones(length - 1, dtype=torch.int32))
        # And the run reaches the batch as one ascending stretch: its rows land
        # adjacent, in step order, wherever the shuffle placed it.
        position = int((order == start).nonzero()[0])
        assert order[position : position + length].tolist() == list(range(start, start + length))


def test_a_run_length_of_one_is_the_independent_row_shuffle() -> None:
    """The default must leave a queued run's batches alone. At a run length of one
    the sampler is `torch.randperm` on the same generator -- one draw per epoch,
    the same order -- rather than a lookalike that shifts gradient statistics."""
    trainer = _load_trainer()
    episode_index, _ = _synthetic_metadata(_RUN_SPANS)
    rows = sum(_RUN_SPANS)
    starts, lengths = trainer._run_blocks(episode_index, 1)
    generator = torch.Generator(device="cpu").manual_seed(7)
    reference = torch.Generator(device="cpu").manual_seed(7)

    assert lengths.tolist() == [1] * rows
    for _ in range(3):
        assert torch.equal(
            trainer._run_epoch_order(starts, lengths, generator),
            torch.randperm(rows, generator=reference),
        )


def test_the_staged_step_is_the_step_the_archive_recorded(dataset_dir: Path) -> None:
    """`step` is derived from row order, so it has to agree with the step every
    stored observation carries; that agreement is what makes the derivation a fact
    about the archive rather than an assumption about how it was written."""
    trainer = _load_trainer()
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    held_out = max(int(entry["seed"]) for entry in manifest["episodes"])
    kept = [entry for entry in manifest["episodes"] if int(entry["seed"]) != held_out]
    recorded: list[int] = []
    episodes: list[int] = []
    for position, entry in enumerate(kept):
        with np.load(dataset_dir / entry["file"]) as archive:
            raw = json.loads(zlib.decompress(archive["raw_json_zlib"].tobytes()))
        recorded.extend(int(item["observation"]["step"]) for item in raw["observations"])
        episodes.extend([position] * len(raw["observations"]))

    train_split, _, _ = trainer.load_dataset(
        [dataset_dir],
        architecture=CONV_ENTITY,
        holdout_seeds=1,
        encode_workers=1,
    )

    assert recorded == list(range(EPISODE_STEPS - 1)) * len(kept)
    assert train_split.staged["step"].tolist() == recorded
    assert train_split.staged["episode_index"].tolist() == episodes


def test_row_metadata_is_dense_per_split_after_a_cap_and_a_holdout(
    dataset_dir: Path, tmp_path: Path
) -> None:
    """Pairing is keyed on `episode_index`, so it must index the split it is
    staged in: the cap drops seeds and the holdout takes whole seeds off the top,
    and an index left over from the uncapped corpus would leave gaps that make
    two unrelated rows -- or none at all -- look like one episode's steps."""
    trainer = _load_trainer()
    directory = _copy_dataset(dataset_dir, tmp_path / "three-seeds")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    lowest = min(int(entry["seed"]) for entry in manifest["episodes"])
    highest = max(int(entry["seed"]) for entry in manifest["episodes"]) + 1
    for seat in (0, 1):
        shutil.copyfile(
            directory / f"episode-{lowest:08d}-seat{seat}.npz",
            directory / f"episode-{highest:08d}-seat{seat}.npz",
        )
        manifest["episodes"].append(
            {"file": f"episode-{highest:08d}-seat{seat}.npz", "seed": highest, "seat": seat}
        )
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    train_split, holdout_split, _ = trainer.load_dataset(
        [directory],
        architecture=CONV_ENTITY,
        holdout_seeds=1,
        encode_workers=1,
        seeds_per_dataset=2,
    )

    per_episode = EPISODE_STEPS - 1
    # The capped seed is gone from both sides and the held-out seed's two seats
    # are the whole holdout, so each split holds exactly two episode-seats.
    for split in (train_split, holdout_split):
        assert split.rows == 2 * per_episode
        assert split.staged["episode_index"].dtype == torch.int32
        assert split.staged["step"].dtype == torch.int32
        assert split.staged["episode_index"].tolist() == [0] * per_episode + [1] * per_episode
        assert split.staged["step"].tolist() == list(range(per_episode)) * 2


def test_the_artifact_records_how_its_batches_were_built(dataset_dir: Path, tmp_path: Path) -> None:
    """Two clones trained with different samplers are different experiments, so
    the sampler's configuration belongs in the provenance the artifact carries."""
    trainer = _load_trainer()
    output = tmp_path / "run"
    arguments = {
        "dataset_dirs": [dataset_dir],
        "output_dir": output,
        "architecture": CONV_ENTITY,
        "config": _tiny_config(),
        "holdout_seeds": 1,
        "epochs": 1,
        "patience": 1,
        "batch_size": 8,
        "run_length": 3,
        "matrix_learning_rate": 1e-3,
        "matrix_weight_decay": 0.0,
        "adam_learning_rate_ratio": 0.35,
        "adam_weight_decay": 0.0,
        "seed": 0,
        "device": torch.device("cpu"),
        "encode_workers": 1,
    }

    trainer.train(**arguments)

    _, payload = load_actor_artifact(output / "bc-actor.pt")
    provenance = payload["bc_provenance"]
    assert (provenance["run_length"], provenance["batch_size"]) == (3, 8)
    with pytest.raises(ValueError, match="must be positive"):
        trainer.train(**{**arguments, "output_dir": tmp_path / "other", "run_length": 0})


def test_the_clone_rate_holds_flat_then_decays_linearly_to_a_floor() -> None:
    """The reference's trapezoid, which is neither a cosine nor a decay to zero."""

    trainer = _load_trainer()
    total = 1000
    cooldown_start = int(total * (1.0 - trainer.COOLDOWN_FRACTION))
    assert trainer._rate_fraction(0, total) == 1.0
    assert trainer._rate_fraction(cooldown_start - 1, total) == 1.0
    assert trainer._rate_fraction(cooldown_start, total) == 1.0
    assert trainer._rate_fraction(total, total) == pytest.approx(trainer.FINAL_RATE_FRACTION)
    tail = [trainer._rate_fraction(step, total) for step in range(cooldown_start, total + 1)]
    assert all(later <= earlier for earlier, later in itertools.pairwise(tail))
    # Linear: the middle of the cooldown is the middle of the range. A cosine
    # would sit well above this.
    middle = trainer._rate_fraction((cooldown_start + total) // 2, total)
    assert middle == pytest.approx((1.0 + trainer.FINAL_RATE_FRACTION) / 2, abs=2e-3)


def test_the_nesterov_coefficient_warms_up_holds_and_cools_back_down() -> None:
    trainer = _load_trainer()
    total = 4000
    assert trainer._momentum_at(0, total) == pytest.approx(trainer.MOMENTUM_MINIMUM)
    warmed = trainer._momentum_at(trainer.MOMENTUM_WARMUP_STEPS, total)
    assert warmed == pytest.approx(trainer.MOMENTUM_MAXIMUM)
    assert trainer._momentum_at(total // 2, total) == pytest.approx(trainer.MOMENTUM_MAXIMUM)
    assert trainer._momentum_at(total - 1, total) < trainer.MOMENTUM_MAXIMUM
    assert trainer._momentum_at(total, total) == pytest.approx(trainer.MOMENTUM_MINIMUM)


def test_a_short_clone_still_reaches_the_held_coefficient() -> None:
    """An uncapped 300-step warmup would span a short run and never arrive."""

    trainer = _load_trainer()
    total = 40
    assert total < trainer.MOMENTUM_WARMUP_STEPS
    assert trainer._momentum_at(total // 2, total) == pytest.approx(trainer.MOMENTUM_MAXIMUM)


def test_the_schedule_sets_every_group_from_its_own_base_rate() -> None:
    trainer = _load_trainer()
    matrix = torch.nn.Parameter(torch.randn(16, 16))
    vector = torch.nn.Parameter(torch.randn(16))
    optimizer = NorMuon(
        [matrix], [vector], learning_rate=3e-3, adam_learning_rate=1e-3, weight_decay=1.2
    )
    total = 100
    trainer._apply_schedule(optimizer, total, total)
    groups = {group["kind"]: group for group in optimizer.param_groups}
    assert set(groups) == {"normuon", "adam"}
    for group in optimizer.param_groups:
        expected = group["base_lr"] * trainer.FINAL_RATE_FRACTION
        assert group["lr"] == pytest.approx(expected)
    assert groups["normuon"]["momentum"] == pytest.approx(trainer.MOMENTUM_MINIMUM)
    # The Adam half has no Nesterov coefficient to schedule, and inventing one
    # here would be read by nothing.
    assert "momentum" not in groups["adam"]


def test_a_typo_in_the_compile_mode_fails_before_the_corpus_is_staged(tmp_path: Path) -> None:
    """Staging takes minutes, so a bad mode must not surface at the first minibatch."""
    trainer = _load_trainer()
    with pytest.raises(ValueError, match="unknown compile mode"):
        trainer.train(
            dataset_dirs=[tmp_path],
            output_dir=tmp_path / "run",
            architecture=CONV_ENTITY,
            config=_tiny_config(),
            holdout_seeds=1,
            epochs=1,
            patience=1,
            batch_size=8,
            compile_mode="max-autotune-no-cudagrahps",
            matrix_learning_rate=1e-3,
            matrix_weight_decay=1.2,
            adam_learning_rate_ratio=0.35,
            adam_weight_decay=0.005,
            seed=0,
            device=torch.device("cpu"),
            encode_workers=1,
        )


def test_the_shipped_compile_default_is_a_real_inductor_mode(monkeypatch, tmp_path: Path) -> None:
    """The CLI default and the function default deliberately differ; pin both.

    `train`'s default is `none` so the CPU suite never pays a compilation, while
    the CLI default is the measured mode so a production run gets the 2.03x
    without being asked. That split is a trap unless it is pinned: the shipped
    value has to be a mode inductor actually accepts, and it must not be a
    cudagraphs mode -- an epoch's last minibatch is a short tail, so the shape
    varies and a captured graph would not fit it.
    """
    trainer = _load_trainer()
    monkeypatch.setattr(
        sys, "argv", ["train_bc.py", "--dataset", str(tmp_path), "--output", str(tmp_path / "run")]
    )
    shipped = trainer.parse_args().compile_mode
    assert shipped == "default"
    assert shipped in trainer.COMPILE_MODES
    # Read from inductor's config rather than the mode's name: `reduce-overhead`
    # does not say "cudagraphs" and enables them anyway.
    assert not torch._inductor.list_mode_options(shipped).get("triton.cudagraphs")
    assert inspect.signature(trainer.train).parameters["compile_mode"].default == "none"


def test_a_cudagraphs_mode_is_refused(tmp_path: Path) -> None:
    """`reduce-overhead` enables cudagraphs without saying so in its name.

    An epoch's last minibatch is a short tail, so a captured graph would either
    recapture per shape or fail. The refusal reads inductor's config for the
    mode, which is why a name-based check would have let this one through.
    """
    trainer = _load_trainer()
    with pytest.raises(ValueError, match="enables cudagraphs"):
        trainer.train(
            dataset_dirs=[tmp_path],
            output_dir=tmp_path / "run",
            architecture=CONV_ENTITY,
            config=_tiny_config(),
            holdout_seeds=1,
            epochs=1,
            patience=1,
            batch_size=8,
            compile_mode="reduce-overhead",
            matrix_learning_rate=1e-3,
            matrix_weight_decay=1.2,
            adam_learning_rate_ratio=0.35,
            adam_weight_decay=0.005,
            seed=0,
            device=torch.device("cpu"),
            encode_workers=1,
        )


def test_the_latent_auxiliary_trains_and_is_journalled(dataset_dir: Path, tmp_path: Path) -> None:
    """The auxiliary has to move p_psi and appear in the journal.

    A term that is computed and thrown away looks identical to one that works,
    so this pins the observable: the aux scalars exist, the eligible-pair count
    is positive, and the belief has not collapsed.
    """
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
        batch_size=64,
        run_length=4,
        latent_dynamics_coefficient=1.0,
        latent_decode_coefficient=0.5,
        latent_horizon=2,
        matrix_learning_rate=1e-3,
        matrix_weight_decay=1.2,
        adam_learning_rate_ratio=0.35,
        adam_weight_decay=0.005,
        seed=0,
        device=torch.device("cpu"),
        encode_workers=1,
    )
    record = json.loads((output / "metrics.jsonl").read_text().splitlines()[0])
    for name in ("latent_dynamics", "latent_decode", "latent_unit_half", "latent_market_half"):
        assert record[name] > 0.0, name
    assert record["latent_eligible"] > 0.0
    # A collapsed belief drives the dynamics term to zero for free; the
    # stop-gradient exists to prevent it, and this is the reading that shows it.
    assert record["belief_dispersion"] > 0.0
    payload = torch.load(output / "bc-actor.pt", map_location="cpu", weights_only=False)
    assert payload["bc_provenance"]["latent_dynamics_coefficient"] == 1.0
    assert payload["bc_provenance"]["latent_horizon"] == 2
    # p_psi is training-only: it must never reach an actor artifact, which
    # inference, league snapshots and the frozen-ensemble stack all load whole.
    assert not any(name.startswith("predictor") for name in payload["actor"])


def test_the_auxiliary_is_refused_when_the_sampler_gives_it_no_pairs(
    dataset_dir: Path, tmp_path: Path
) -> None:
    """An all-false eligibility mask reports a loss of exactly zero.

    That reads as a converged auxiliary rather than an absent one, so the
    combination is refused instead of trained.
    """
    trainer = _load_trainer()
    with pytest.raises(ValueError, match="needs --run-length above it"):
        trainer.train(
            dataset_dirs=[dataset_dir],
            output_dir=tmp_path / "run",
            architecture=CONV_ENTITY,
            config=_tiny_config(),
            holdout_seeds=1,
            epochs=1,
            patience=1,
            batch_size=64,
            run_length=1,
            latent_dynamics_coefficient=1.0,
            latent_horizon=1,
            matrix_learning_rate=1e-3,
            matrix_weight_decay=1.2,
            adam_learning_rate_ratio=0.35,
            adam_weight_decay=0.005,
            seed=0,
            device=torch.device("cpu"),
            encode_workers=1,
        )
