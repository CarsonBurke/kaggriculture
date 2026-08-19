from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import torch

from kaggriculture.actions import MarketKind
from kaggriculture.inference import (
    ACTOR_ARTIFACT_FORMAT_VERSION,
    CHECKPOINT_FORMAT_VERSION,
    LEGACY_CHECKPOINT_FORMAT_VERSIONS,
    actor_artifact_from_checkpoint,
    load_actor_artifact,
)
from kaggriculture.model import FarmActor, ModelConfig
from kaggriculture.provenance import (
    is_legacy_run_provenance,
    run_provenance_from_decision,
    source_identity,
)


@pytest.mark.parametrize(
    "checkpoint_version",
    [
        ACTOR_ARTIFACT_FORMAT_VERSION,
        *sorted(LEGACY_CHECKPOINT_FORMAT_VERSIONS),
        CHECKPOINT_FORMAT_VERSION,
    ],
)
def test_actor_artifact_round_trip(tmp_path: Path, checkpoint_version: int) -> None:
    config = ModelConfig(
        cnn_width=16, cnn_blocks=1, model_dim=32, transformer_layers=3, attention_heads=4
    )
    actor = FarmActor(config)
    artifact = actor_artifact_from_checkpoint(
        {
            "format_version": checkpoint_version,
            "model_config": config.to_dict(),
            "actor": actor.state_dict(),
            "iteration": 3,
            "source_identity": source_identity(),
        }
    )
    path = tmp_path / "model.pt"
    torch.save(artifact, path)

    restored, metadata = load_actor_artifact(path)

    assert metadata["iteration"] == 3
    assert metadata["format_version"] == ACTOR_ARTIFACT_FORMAT_VERSION
    for expected, actual in zip(actor.parameters(), restored.parameters(), strict=True):
        assert torch.equal(expected, actual)


@pytest.mark.parametrize(
    "checkpoint_version",
    [*sorted(LEGACY_CHECKPOINT_FORMAT_VERSIONS), CHECKPOINT_FORMAT_VERSION],
)
def test_full_checkpoint_loads_directly_as_actor(tmp_path: Path, checkpoint_version: int) -> None:
    config = ModelConfig(
        cnn_width=8,
        cnn_blocks=1,
        model_dim=16,
        transformer_layers=3,
        attention_heads=2,
    )
    actor = FarmActor(config)
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format_version": checkpoint_version,
            "model_config": config.to_dict(),
            "actor": actor.state_dict(),
            "source_identity": source_identity(),
        },
        path,
    )

    restored, metadata = load_actor_artifact(path)

    assert metadata["format_version"] == checkpoint_version
    for expected, actual in zip(actor.parameters(), restored.parameters(), strict=True):
        assert torch.equal(expected, actual)


@pytest.mark.parametrize("calibrated", [False, True])
def test_submission_bundle_is_isolated_complete_and_within_action_timeout(
    tmp_path: Path,
    calibrated: bool,
) -> None:
    config = ModelConfig()
    actor = FarmActor(config)
    with torch.no_grad():
        actor.market_kind.weight.zero_()
        actor.market_kind.bias.fill_(-50.0)
        actor.market_kind.bias[MarketKind.BUY_SEED_WHEAT] = 50.0
        actor.market_quantity_context.weight.zero_()
        actor.market_quantity_bias.fill_(-50.0)
        actor.market_quantity_bias[:, -1] = 50.0
    checkpoint = tmp_path / "checkpoint.pt"
    archive = tmp_path / "submission.tar.gz"
    run_provenance = None
    if calibrated:
        run_provenance = run_provenance_from_decision(
            {
                "source_identity": source_identity(),
                "rollout_forward_mode": "eager",
                "update_compile_mode": "eager",
                "eager_report_sha256": "a" * 64,
                "eager_report_size_bytes": 100,
                "mixed_report_sha256": "c" * 64,
                "mixed_report_size_bytes": 110,
                "compiled_report_sha256": "b" * 64,
                "compiled_report_size_bytes": 120,
                "minimum_compile_speedup": 1.05,
                "attributed_knob_speedups": {
                    "rollout_forward_mode": 1.0,
                    "update_compile_mode": 1.0,
                },
            },
        )
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_config": config.to_dict(),
            "actor": actor.state_dict(),
            "iteration": 17,
            "source_identity": source_identity(),
            "run_provenance": run_provenance,
        },
        checkpoint,
    )
    repository = Path(__file__).parents[1]
    evaluation = tmp_path / "evaluation.json"
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    opponent_provenance = {
        "kind": "python_file",
        "path": "/var/tmp/public-v27.py",
        "sha256": "c" * 64,
        "size_bytes": 100,
    }
    evaluation.write_text(
        json.dumps(
            {
                "valid_for_selection": True,
                "opponent_label": "public-v27",
                "seed_count": 128,
                "seed_start": 20_000_000,
                "paired_seats": True,
                "summary": {"score_rate": 0.75},
                "selection_provenance": {
                    "best_output_sha256": checkpoint_digest,
                    "sha256": "d" * 64,
                    "run_provenance": run_provenance,
                    "screening_seed_start": 10_000_000,
                    "screening_seed_count": 32,
                    "opponent_provenance": {
                        "public-v27": {
                            "kind": "python_file",
                            "sha256": "c" * 64,
                            "size_bytes": 100,
                        }
                    },
                },
                "opponent_provenance": opponent_provenance,
                "artifact_provenance": {
                    "sha256": checkpoint_digest,
                    "source_identity": source_identity(),
                    "run_provenance": run_provenance,
                },
            }
        ),
        encoding="utf-8",
    )
    starter_evaluation = tmp_path / "starter.json"
    starter_evaluation.write_text(
        json.dumps(
            {
                "valid_for_selection": True,
                "opponent_label": "starter",
                "seed_count": 64,
                "paired_seats": True,
                "summary": {"score_rate": 1.0},
                "artifact_provenance": {
                    "sha256": checkpoint_digest,
                    "source_identity": source_identity(),
                    "run_provenance": run_provenance,
                },
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "build_submission.py"),
            "--checkpoint",
            str(checkpoint),
            "--evaluation-report",
            str(evaluation),
            "--builtin-evaluation-report",
            str(starter_evaluation),
            "--output",
            str(archive),
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    required = {
        "main.py",
        "model.pt",
        "evaluation.json",
        "manifest.json",
        "kaggriculture/__init__.py",
        "kaggriculture/actions.py",
        "kaggriculture/constants.py",
        "kaggriculture/encoding.py",
        "kaggriculture/inference.py",
        "kaggriculture/model.py",
        "kaggriculture/policy.py",
        "kaggriculture/provenance.py",
        "kaggriculture/registry.py",
        "kaggriculture/structured.py",
        "kaggriculture/tokens.py",
    }
    with tarfile.open(archive, "r:gz") as bundle:
        assert set(bundle.getnames()) == required
        bundle.extractall(tmp_path / "extracted", filter="data")
    # Keep the actor-only artifact compact without constraining worthwhile
    # policy capacity to the former CNN's incidental five-megabyte footprint.
    assert archive.stat().st_size < 8_000_000

    probe = r"""
import json
import sys
import time
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
import kaggriculture
from kaggle_environments import make
from kaggle_environments.agent import get_last_callable

assert Path(kaggriculture.__file__).resolve().is_relative_to(root)
main_path = root / "main.py"
raw_agent = get_last_callable(main_path.read_text(encoding="utf-8"), path=str(main_path))
environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 991})
observation = environment.reset(2)[0].observation
farm = observation["farms"][0]
farm["hands"] = [[4 + index % 2, 4 + index // 2 % 2] for index in range(15)]
farm["money"] = 1_000_000_000
farm["hires_today"] = 15
observation["private"]["inventories"] = [{} for _ in range(16)]

elapsed = []
for _ in range(5):
    started = time.perf_counter()
    action = raw_agent(observation)
    elapsed.append(time.perf_counter() - started)
assert len(action["hands"]) == 15
assert len(action["market"]) == 10
assert action["market"] == [["BUY_SEED", "WHEAT", 100]] * 10
assert max(elapsed) < 1.0
print(json.dumps({"max_action_seconds": max(elapsed), "action": action}))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(tmp_path / "extracted")],
        cwd=tmp_path / "extracted",
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Kaggle's import path may log framework diagnostics to stdout before the
    # probe result, so treat the final non-empty line as the machine payload.
    result = json.loads([line for line in completed.stdout.splitlines() if line][-1])
    assert result["max_action_seconds"] < 1.0


def _build_submission_module():
    path = Path(__file__).parents[1] / "scripts" / "build_submission.py"
    spec = importlib.util.spec_from_file_location("kaggriculture_build_submission", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _submission_inputs(tmp_path: Path) -> tuple[Path, dict, dict]:
    """A checkpoint and the two reports a submission needs, all mutually bound."""
    config = ModelConfig()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_config": config.to_dict(),
            "actor": FarmActor(config).state_dict(),
            "iteration": 5,
            "source_identity": source_identity(),
            "run_provenance": None,
        },
        checkpoint,
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    provenance = {
        "sha256": digest,
        "source_identity": source_identity(),
        "run_provenance": None,
    }
    finalist = {
        "valid_for_selection": True,
        "opponent_label": "public-v27",
        "seed_count": 128,
        "seed_start": 20_000_000,
        "paired_seats": True,
        "summary": {"score_rate": 0.75},
        "selection_provenance": {
            "best_output_sha256": digest,
            "sha256": "d" * 64,
            "run_provenance": None,
            "screening_seed_start": 10_000_000,
            "screening_seed_count": 32,
            "opponent_provenance": {
                "public-v27": {"kind": "python_file", "sha256": "c" * 64, "size_bytes": 100}
            },
        },
        "opponent_provenance": {
            "kind": "python_file",
            "path": "/var/tmp/public-v27.py",
            "sha256": "c" * 64,
            "size_bytes": 100,
        },
        "artifact_provenance": provenance,
    }
    starter = {
        "valid_for_selection": True,
        "opponent_label": "starter",
        "seed_count": 64,
        "paired_seats": True,
        "summary": {"score_rate": 1.0},
        "artifact_provenance": provenance,
    }
    return checkpoint, finalist, starter


def test_submission_refuses_an_agent_that_loses(tmp_path: Path) -> None:
    """The provenance gate cannot see strength, and both shipped finalists prove it.

    `evaluations/vapo-lv2-iter415-finalist-v27.json` and its `vapo-main` sibling
    each record `score_rate` 0.0 with 0 wins over 256 seats while stamped
    `valid_for_selection: True`. That flag means the evaluation ran, so a gate
    reading only provenance packaged agents that never won a game.
    """
    build_submission = _build_submission_module()
    checkpoint, finalist, starter = _submission_inputs(tmp_path)
    finalist_path = tmp_path / "finalist.json"
    starter_path = tmp_path / "starter.json"
    output = tmp_path / "submission.tar.gz"

    def write(finalist_payload: dict, starter_payload: dict) -> None:
        finalist_path.write_text(json.dumps(finalist_payload), encoding="utf-8")
        starter_path.write_text(json.dumps(starter_payload), encoding="utf-8")

    def attempt(**overrides) -> dict:
        return build_submission.build(
            checkpoint,
            finalist_path,
            output,
            builtin_evaluation_reports=[starter_path],
            minimum_score_rate=0.5,
            minimum_builtin_score_rate=0.9,
            minimum_builtin_seed_count=32,
            **overrides,
        )

    write(finalist, starter)
    manifest = attempt()
    assert manifest["evaluation"]["score_rate"] == 0.75
    assert manifest["strength_gate"]["builtin_score_rates"] == {"starter": 1.0}

    # The exact historical failure: every seat lost, provenance immaculate.
    write({**finalist, "summary": {"score_rate": 0.0}}, starter)
    with pytest.raises(ValueError, match="below the required"):
        attempt()

    # Losing to the carrot-loop heuristic must block a submission on its own,
    # even when the public-v27 number is healthy.
    write(finalist, {**starter, "summary": {"score_rate": 0.4}})
    with pytest.raises(ValueError, match="against the built-in starter"):
        attempt()

    # A tiny sample is not evidence of beating it.
    write(finalist, {**starter, "seed_count": 4})
    with pytest.raises(ValueError, match="seed clusters, below the required"):
        attempt()

    # An aborted evaluation writes a null rate, which must not read as zero or crash.
    write({**finalist, "summary": {"score_rate": None}}, starter)
    with pytest.raises(ValueError, match="no finite score rate"):
        attempt()

    write(finalist, starter)
    with pytest.raises(ValueError, match="requires an evaluation against the built-in starter"):
        build_submission.build(
            checkpoint,
            finalist_path,
            output,
            builtin_evaluation_reports=[],
            minimum_score_rate=0.5,
        )


def test_weights_load_across_the_provenance_bump_but_do_not_export(tmp_path: Path) -> None:
    """Loading weights and carrying a calibration claim forward are different
    operations, and only the second needs the claim to be interpretable.

    `load_actor_artifact` is the read path for deliberately cross-tree work --
    `--init-actor-from`, replay viewing, behavior audits -- none of which reads
    run provenance. Refusing those over a pre-split calibration record would
    reject good weights for a field the caller never touches. Export is the
    opposite case: it copies provenance into a submission, where an
    uninterpretable claim would be asserted as though it were recoverable.
    """
    config = ModelConfig(cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3)
    actor = FarmActor(config)
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_config": config.to_dict(),
        "actor": actor.state_dict(),
        "source_identity": source_identity(),
        # A well-formed record from before the rollout/update split.
        "run_provenance": {"format_version": 1, "sha256": "a" * 64, "calibration": {}},
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)

    restored, payload = load_actor_artifact(path)
    for expected, actual in zip(actor.parameters(), restored.parameters(), strict=True):
        assert torch.equal(expected, actual)
    # The stale record is dropped from the loader's view, not rewritten on disk.
    assert payload["run_provenance"]["format_version"] == 1

    # Version 2 is dropped on the same read path for a different reason: it did
    # split the decision per knob, but its per-phase speedups were differenced
    # across a pair of runs that moved both knobs at once, so each phase's ratio
    # carries whatever drift that pair happened to have. The configuration that
    # isolates a single knob was never run, so those numbers cannot be
    # re-attributed after the fact and the record is rejected, not migrated.

    with pytest.raises(ValueError, match="carries superseded calibration provenance"):
        actor_artifact_from_checkpoint(checkpoint)
    # Every superseded version is refused at the same boundary, so a checkpoint
    # written by an earlier tree cannot export a decision whose evidence no
    # longer substantiates it. Version 3's rollout knob is a boolean whose
    # only "on" value was `cudagraphs`, so the speedup it certifies belongs to a
    # mode measurement rejects -- 5.309 ms against eager's 4.907 ms on the
    # isolated collection forward -- and no mode can be recovered from `False`.
    # Version 4's update knob is the same defect one phase over: its `true`
    # names all four compiled modes at once, so nothing in the record says which
    # one the speedup beside it was measured under.
    for superseded in (2, 3, 4):
        assert is_legacy_run_provenance(
            {"format_version": superseded, "sha256": "a" * 64, "calibration": {}}
        )
        with pytest.raises(ValueError, match="carries superseded calibration provenance"):
            actor_artifact_from_checkpoint(
                {
                    **checkpoint,
                    "run_provenance": {
                        "format_version": superseded,
                        "sha256": "a" * 64,
                        "calibration": {},
                    },
                }
            )

    # Corruption is not leniency's business: only a well-formed older version
    # is dropped, and anything else still raises on the read path.
    torch.save({**checkpoint, "run_provenance": {"nonsense": True}}, path)
    with pytest.raises(ValueError, match="invalid schema"):
        load_actor_artifact(path)


@pytest.mark.parametrize("version", [None, 1, 2, 3, 4])
def test_actor_artifact_rejects_incompatible_format(tmp_path: Path, version) -> None:
    path = tmp_path / "model.pt"
    torch.save({"format_version": version}, path)

    with pytest.raises(ValueError, match="unsupported actor artifact format"):
        load_actor_artifact(path)


def _bundle(root: Path, main_source: str) -> Path:
    """Assemble a submission bundle by hand, so `main.py` can be made faulty."""
    builder = _load_build_submission()
    package = root / "kaggriculture"
    package.mkdir(parents=True)
    (root / "main.py").write_text(main_source, encoding="utf-8")
    config = ModelConfig()
    actor = FarmActor(config)
    with torch.no_grad():
        actor.market_kind.weight.zero_()
        actor.market_kind.bias.fill_(-50.0)
        actor.market_kind.bias[MarketKind.BUY_SEED_WHEAT] = 50.0
        actor.market_quantity_context.weight.zero_()
        actor.market_quantity_bias.fill_(-50.0)
        actor.market_quantity_bias[:, -1] = 50.0
    torch.save(
        {
            "format_version": ACTOR_ARTIFACT_FORMAT_VERSION,
            "model_config": config.to_dict(),
            "actor": actor.state_dict(),
            "iteration": 3,
            "source_identity": source_identity(),
            "run_provenance": None,
        },
        root / "model.pt",
    )
    for name in builder.PACKAGE_FILES:
        shutil.copy2(Path(__file__).parents[1] / "src" / "kaggriculture" / name, package / name)
    return root


def _load_build_submission():
    spec = importlib.util.spec_from_file_location(
        "build_submission_under_test",
        Path(__file__).parents[1] / "scripts" / "build_submission.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_the_bundle_smoke_test_refuses_an_agent_that_never_acts(tmp_path: Path) -> None:
    # The expensive failure is quiet: `kaggle_environments` swallows an agent
    # exception, seats PASS for the rest of the episode, and reports DONE with the
    # starting bank intact. On the leaderboard that is indistinguishable from a
    # merely weak submission, and it costs a day of submission budget to learn.
    builder = _load_build_submission()

    playing = _bundle(tmp_path / "playing", builder.MAIN)
    result = builder._smoke_test(playing)
    assert result["status"] == "DONE"
    assert result["submitted"] == builder._SMOKE_STEPS - 1
    assert result["acting"] > 0

    with pytest.raises(ValueError, match="passed on every step"):
        builder._smoke_test(_bundle(tmp_path / "silent", "def agent(obs):\n    return {}\n"))
    with pytest.raises(ValueError, match="failed to run"):
        builder._smoke_test(
            _bundle(tmp_path / "raising", "def agent(obs):\n    raise ValueError('bad')\n")
        )
    # The trap the entrypoint template exists to avoid: `getfullargspec` counts
    # `self`, so a callable object is invoked with two arguments, the TypeError is
    # swallowed, and the seat submits nothing for the whole episode.
    with pytest.raises(ValueError, match="failed to run"):
        builder._smoke_test(
            _bundle(
                tmp_path / "callable",
                "from pathlib import Path\n"
                "import kaggriculture\n"
                "from kaggriculture.inference import CheckpointAgent\n"
                "agent = CheckpointAgent(\n"
                "    Path(kaggriculture.__file__).resolve().parent.parent / 'model.pt'\n"
                ")\n",
            )
        )


def _load_build_plan_submission():
    spec = importlib.util.spec_from_file_location(
        "build_plan_submission_under_test",
        Path(__file__).parents[1] / "scripts" / "build_plan_submission.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_the_plan_patch_cancels_exactly_the_named_steps(tmp_path: Path) -> None:
    # The failure mode is silent and expensive: a patch that emits
    # `frozenset(220, 224)` raises at import, and one that emits a truthy scalar
    # would cancel the market on every step. Either ships an agent that is not the
    # one the search measured, and the archive still looks well formed.
    builder = _load_build_plan_submission()
    source = tmp_path / "plan.py"
    source.write_text(
        "def _get(obs, key, default=None):\n"
        "    return obs.get(key, default)\n"
        "\n"
        "\n"
        "def agent(obs, configuration=None):\n"
        "    return {'farmer': ['PASS'], 'hands': [], 'market': [['SELL', 'WHEAT', 1]]}\n",
        encoding="utf-8",
    )
    patched = tmp_path / "main.py"
    patched.write_text(builder.patched_source(source, (2, 5)), encoding="utf-8")

    module = builder._load(patched, "patched_plan_under_test")
    assert frozenset({2, 5}) == module._CANCELLED_MARKET_STEPS
    cancelled = [step for step in range(8) if module.agent({"step": step})["market"] == []]
    assert cancelled == [2, 5]
    # The rest of the action is the plan's own, so the edit cannot be credited
    # with a change it did not make.
    assert module.agent({"step": 3})["market"] == [["SELL", "WHEAT", 1]]

    with pytest.raises(SystemExit, match="already patched"):
        builder.patched_source(patched, (2, 5))
    bare = tmp_path / "bare.py"
    bare.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no top-level agent"):
        builder.patched_source(bare, (2,))


def test_the_plan_patch_shrinks_only_priced_purchases(tmp_path: Path) -> None:
    # Reducing a sell would stop the plan banking its harvest, and reducing a bare
    # `HIRE` would corrupt an order the engine reads positionally. Both are silent:
    # the agent still returns a well formed action and still plays 720 steps.
    builder = _load_build_plan_submission()
    source = tmp_path / "plan.py"
    source.write_text(
        "def _get(obs, key, default=None):\n"
        "    return obs.get(key, default)\n"
        "\n"
        "\n"
        "def agent(obs, configuration=None):\n"
        "    return {\n"
        "        'farmer': ['PASS'],\n"
        "        'hands': [],\n"
        "        'market': [\n"
        "            ['HIRE'],\n"
        "            ['BUY_SEED', 'MELON', 7],\n"
        "            ['SELL', 'WHEAT', 9],\n"
        "            ['BUY_PRODUCT', 'WHEAT', 1],\n"
        "        ],\n"
        "    }\n",
        encoding="utf-8",
    )
    patched = tmp_path / "main.py"
    patched.write_text(builder.patched_source(source, (5,), ((3, 2),)), encoding="utf-8")

    module = builder._load(patched, "reduced_plan_under_test")
    assert module._REDUCED_BUY_STEPS == {3: 2}
    assert module.agent({"step": 3})["market"] == [
        ["HIRE"],
        ["BUY_SEED", "MELON", 5],
        ["SELL", "WHEAT", 9],
        # Floored at one: the engine has no smaller order, so the alternative to a
        # shrink this deep is a cancellation, which is a different edit.
        ["BUY_PRODUCT", "WHEAT", 1],
    ]
    assert module.agent({"step": 4})["market"] == [
        ["HIRE"],
        ["BUY_SEED", "MELON", 7],
        ["SELL", "WHEAT", 9],
        ["BUY_PRODUCT", "WHEAT", 1],
    ]
    assert module.agent({"step": 5})["market"] == []

    with pytest.raises(SystemExit, match="both cancelled and reduced"):
        builder.patched_source(source, (3,), ((3, 2),))
