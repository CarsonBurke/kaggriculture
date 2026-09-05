from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kaggriculture.inference import CHECKPOINT_FORMAT_VERSION
from kaggriculture.opponents import normalize_opponent
from kaggriculture.orientation import Orientation
from kaggriculture.provenance import source_identity
from kaggriculture.structured import StructuredActor, StructuredConfig


def _script():
    path = Path(__file__).parents[1] / "scripts" / "external_eval_worker.py"
    spec = importlib.util.spec_from_file_location("kaggriculture_external_eval_worker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Python 3.13 dataclass creation requires the defining module in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_builtin_opponents_pass_through_and_missing_files_fail(tmp_path: Path) -> None:
    assert normalize_opponent("starter") == ("starter", "starter")

    agent = tmp_path / "agent.py"
    agent.write_text("def act(observation):\n    return {}\n")
    label, runnable = normalize_opponent(str(agent))
    assert label == "agent.py"
    assert runnable == str(agent)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        normalize_opponent(str(tmp_path / "missing.py"))


def test_public_v16_alias_resolves_to_fixed_file(monkeypatch, tmp_path: Path) -> None:
    import kaggriculture.opponents as opponents

    teacher = tmp_path / "v16.py"
    teacher.write_text("def agent(obs): return {}\n", encoding="utf-8")
    monkeypatch.setattr(opponents, "PUBLIC_V16_TEACHER", teacher)

    label, resolved = normalize_opponent("v16")

    assert label == "public-v16"
    assert resolved == str(teacher.resolve())
    assert normalize_opponent("public-v16") == (label, resolved)


def test_game_outcome_scores_wins_ties_and_losses() -> None:
    module = _script()
    win = module.GameOutcome(0, 0, 100.0, 50.0, None)
    tie = module.GameOutcome(0, 1, 70.0, 70.0, None)
    loss = module.GameOutcome(1, 0, 10.0, 20.0, None)
    failed = module.GameOutcome(1, 1, None, None, "seat 0 status is ERROR")

    assert (win.score, tie.score, loss.score) == (1.0, 0.5, 0.0)
    assert win.complete and not failed.complete


def test_evaluate_opponent_summarizes_only_completed_games(monkeypatch, tmp_path: Path) -> None:
    module = _script()
    outcomes = iter(
        [
            module.GameOutcome(0, 0, 100.0, 50.0, None),
            module.GameOutcome(0, 1, 60.0, 60.0, None),
            module.GameOutcome(1, 0, None, None, "environment did not reach DONE"),
            module.GameOutcome(1, 1, 30.0, 40.0, None),
        ]
    )
    monkeypatch.setattr(module, "_play_game", lambda *arguments: next(outcomes))

    record = module.evaluate_opponent(
        object(),
        "starter",
        "starter",
        iteration=40,
        artifact_name="league-actor-00000040.pt",
        artifact_digest="ab" * 32,
        member=None,
        seeds=range(7, 9),
        episode_steps=720,
    )

    assert record["event"] == "external_eval"
    assert record["iteration"] == 40
    assert record["opponent"] == "starter"
    assert record["opponent_sha256"] is None
    assert record["seed_start"] == 7
    assert (record["games"], record["completed_games"]) == (4, 3)
    assert record["money_mean"] == pytest.approx((100.0 + 60.0 + 30.0) / 3)
    assert record["opponent_money_mean"] == pytest.approx(50.0)
    assert record["score_rate"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert record["errors"] == ["environment did not reach DONE"]
    # A single-learner row carries the key with a null, so a reader never has to
    # decide whether an absent key means one learner or an older worker.
    assert record["agent"] is None
    assert record["artifact"] == "league-actor-00000040.pt"


def test_members_refuses_a_spec_a_reader_could_not_deduplicate() -> None:
    module = _script()

    assert module._members("") == [None]
    assert module._members(" ") == [None]
    assert module._members("0,2,1") == [0, 2, 1]
    # Two rows for one member would collide on (iteration, agent, opponent),
    # which is the key a reader deduplicates last-wins on.
    with pytest.raises(ValueError, match="names a member twice"):
        module._members("1,1")
    with pytest.raises(ValueError, match="names a negative member"):
        module._members("-1")
    with pytest.raises(ValueError, match="named no member"):
        module._members(",")


def test_fused_structured_checkpoint_is_evaluated_through_cpu_submission_layout(
    tmp_path: Path,
) -> None:
    module = _script()
    config = StructuredConfig(
        model_dim=128,
        attention_heads=8,
        ffn_multiplier=2,
        farm_blocks=1,
        opponent_latents=4,
        latents=8,
        core_layers=2,
        fused_mlp=True,
    )
    fused = StructuredActor(config)
    artifact = tmp_path / "checkpoint-000012.pt"
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "architecture": "structured",
            "iteration": 12,
            "model_config": config.to_dict(),
            "actor": fused.state_dict(),
            "seed_usage": [],
            "source_identity": source_identity(),
            "run_provenance": None,
        },
        artifact,
    )

    portable, orientation = module._load_member(artifact, None)

    assert isinstance(portable, StructuredActor)
    assert portable.config.fused_mlp is False
    assert next(portable.parameters()).device.type == "cpu"
    assert orientation is Orientation.IDENTITY


def test_fused_population_member_preserves_selected_weights_and_orientation(
    tmp_path: Path,
) -> None:
    module = _script()
    config = StructuredConfig(
        model_dim=128,
        attention_heads=8,
        ffn_multiplier=2,
        farm_blocks=1,
        opponent_latents=4,
        latents=8,
        core_layers=2,
        fused_mlp=True,
    )
    fused = StructuredActor(config)
    first = {name: value.clone() for name, value in fused.state_dict().items()}
    second = {name: value.clone() for name, value in fused.state_dict().items()}
    copied_key = next(
        name
        for name in second
        if not name.endswith(".mlp.up.weight") and not name.endswith(".mlp.down.weight")
    )
    first[copied_key].zero_()
    second[copied_key].fill_(1)
    artifact = tmp_path / "population.pt"
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "architecture": "structured",
            "iteration": 12,
            "model_config": config.to_dict(),
            "agents": [
                {"actor": first, "orientation": int(Orientation.IDENTITY)},
                {"actor": second, "orientation": int(Orientation.MIRROR_X)},
            ],
            "source_identity": source_identity(),
            "run_provenance": None,
            "seed_usage": [],
        },
        artifact,
    )

    portable, orientation = module._load_member(artifact, 1)

    assert portable.config.fused_mlp is False
    assert torch.equal(portable.state_dict()[copied_key], second[copied_key])
    assert orientation is Orientation.MIRROR_X
    malformed = tmp_path / "malformed-population.pt"
    torch.save(
        {
            **torch.load(artifact, map_location="cpu", weights_only=False),
            "source_identity": None,
        },
        malformed,
    )
    with pytest.raises(ValueError, match="source identity"):
        module._load_member(malformed, 1)


def test_opponent_digest_follows_the_runnable_not_the_label(monkeypatch, tmp_path: Path) -> None:
    # An agent file literally named after a built-in must still record its
    # file digest; only true built-ins (runnable == name) omit one.
    module = _script()
    imposter = tmp_path / "starter"
    imposter.write_text("def act(observation):\n    return {}\n")
    label, runnable = normalize_opponent(str(imposter))
    assert label == "starter"
    monkeypatch.setattr(
        module,
        "_play_game",
        lambda *arguments: module.GameOutcome(0, 0, 1.0, 0.0, None),
    )

    record = module.evaluate_opponent(
        object(),
        label,
        runnable,
        iteration=1,
        artifact_name="league-actor-00000001.pt",
        artifact_digest="ab" * 32,
        member=2,
        seeds=range(1),
        episode_steps=720,
    )

    assert isinstance(record["opponent_sha256"], str) and len(record["opponent_sha256"]) == 64


def test_append_record_produces_one_json_line_per_call(tmp_path: Path) -> None:
    module = _script()
    output = tmp_path / "journal" / "metrics-external.jsonl"

    module.append_record(output, {"event": "external_eval", "iteration": 1})
    module.append_record(output, {"event": "external_eval", "iteration": 2})

    lines = output.read_text().splitlines()
    assert [json.loads(line)["iteration"] for line in lines] == [1, 2]


def test_worker_writes_completion_only_after_every_expected_probe(monkeypatch, tmp_path) -> None:
    module = _script()
    artifact = tmp_path / "population.pt"
    torch.save({"seed_usage": []}, artifact)
    output = tmp_path / "metrics-external.jsonl"
    args = SimpleNamespace(
        seeds=2,
        episode_steps=720,
        torch_threads=1,
        opponents="starter,pass",
        agents="0,1",
        artifact=artifact,
        iteration=40,
        output=output,
        seed_start=4_000_000,
    )

    class Actor:
        def eval(self):
            return self

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "normalize_opponent", lambda spec: (spec, spec))
    monkeypatch.setattr(module, "snapshot_sha256", lambda _path: "ab" * 32)
    monkeypatch.setattr(
        module, "_load_member", lambda _artifact, _member: (Actor(), Orientation.IDENTITY)
    )
    monkeypatch.setattr(
        module,
        "evaluate_opponent",
        lambda _agent, label, _runnable, **keywords: {
            "event": "external_eval",
            "iteration": keywords["iteration"],
            "artifact": keywords["artifact_name"],
            "artifact_sha256": keywords["artifact_digest"],
            "agent": keywords["member"],
            "opponent": label,
            "games": 4,
            "completed_games": 4,
        },
    )

    module.main()

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [(row["agent"], row["opponent"]) for row in rows[:-1]] == [
        (0, "starter"),
        (0, "pass"),
        (1, "starter"),
        (1, "pass"),
    ]
    assert rows[-1] == {
        "event": "external_eval_complete",
        "iteration": 40,
        "artifact": "population.pt",
        "artifact_sha256": "ab" * 32,
        "members": [0, 1],
        "opponents": ["starter", "pass"],
        "records": 4,
    }


def test_worker_failure_leaves_rows_without_a_completion_marker(monkeypatch, tmp_path) -> None:
    module = _script()
    artifact = tmp_path / "actor.pt"
    torch.save({"seed_usage": []}, artifact)
    output = tmp_path / "metrics-external.jsonl"
    args = SimpleNamespace(
        seeds=2,
        episode_steps=720,
        torch_threads=1,
        opponents="starter",
        agents="",
        artifact=artifact,
        iteration=9,
        output=output,
        seed_start=4_000_000,
    )

    class Actor:
        def eval(self):
            return self

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "normalize_opponent", lambda spec: (spec, spec))
    monkeypatch.setattr(module, "snapshot_sha256", lambda _path: "cd" * 32)
    monkeypatch.setattr(
        module, "_load_member", lambda _artifact, _member: (Actor(), Orientation.IDENTITY)
    )
    monkeypatch.setattr(
        module,
        "evaluate_opponent",
        lambda *_arguments, **keywords: {
            "event": "external_eval",
            "iteration": keywords["iteration"],
            "artifact": keywords["artifact_name"],
            "artifact_sha256": keywords["artifact_digest"],
            "agent": keywords["member"],
            "opponent": "starter",
            "games": 4,
            "completed_games": 3,
        },
    )

    with pytest.raises(RuntimeError, match="3 of 4 games"):
        module.main()

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["external_eval"]


def test_the_agent_callable_takes_exactly_the_one_argument_the_engine_passes() -> None:
    """`kaggle_environments` decides the call shape by introspection.

    It reads the callable's parameter list and invokes a two-parameter agent as
    `agent(observation, configuration)`. A parameter carrying a default still
    counts, so binding the actor as `actor=actor` to dodge late binding hands the
    configuration dict in as the actor. The TypeError is swallowed under
    `debug=False` and the seat submits nothing for the whole episode while the
    bank sits at its 3000 starting money and the journal reads `score_rate` 0.0 --
    a fault indistinguishable from a policy that chose to idle. Measured exactly
    that against starter, public-v27 and public-v16 before this was pinned.
    """
    import inspect

    module = _script()
    agent = module._agent_for(object(), Orientation.IDENTITY)
    spec = inspect.getfullargspec(agent)

    assert spec.args == ["observation"]
    # A default would be invisible here but visible to the engine's arity count.
    assert not spec.defaults and not spec.kwonlyargs
    assert spec.varargs is None and spec.varkw is None


def test_each_member_gets_its_own_actor_rather_than_the_loops_last() -> None:
    # The factory exists to close over its own scope: N callables built in a loop
    # must not all resolve to the final actor.
    module = _script()
    calls: list[str] = []

    def fake_act_batch(actor, observations, *, deterministic, orientation):
        calls.append(actor)
        return type("Out", (), {"actions": [{"actor": actor}]})()

    module.act_batch = fake_act_batch
    agents = [
        module._agent_for(name, Orientation.IDENTITY) for name in ("first", "second", "third")
    ]
    results = [agent({}) for agent in agents]

    assert [result["actor"] for result in results] == ["first", "second", "third"]
    assert calls == ["first", "second", "third"]


def test_external_evaluation_requires_provenance_bound_actor_not_bare_snapshot(tmp_path) -> None:
    """Training opponents lack exposure provenance; evaluate full checkpoints or BC exports."""
    import torch

    from kaggriculture.league import save_actor_snapshot
    from kaggriculture.model import FarmActor, ModelConfig
    from kaggriculture.provenance import source_identity

    module = _script()
    config = ModelConfig(cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3)
    actor = FarmActor(config)

    snapshot = save_actor_snapshot(tmp_path / "league", actor, 3)
    with pytest.raises(ValueError, match="seed_usage"):
        module._load_member(snapshot.path, None)

    artifact = tmp_path / "bc-actor.pt"
    # Built by the trainer's own writer rather than a hand-rolled dict, so this
    # pins the shape a BC run actually produces instead of one this test invented.
    trainer_path = Path(__file__).parents[1] / "scripts" / "train_bc.py"
    trainer_spec = importlib.util.spec_from_file_location("kaggriculture_train_bc", trainer_path)
    assert trainer_spec is not None and trainer_spec.loader is not None
    trainer = importlib.util.module_from_spec(trainer_spec)
    sys.modules[trainer_spec.name] = trainer
    trainer_spec.loader.exec_module(trainer)
    torch.save(
        trainer._artifact_payload(
            "entity-cnn",
            actor,
            config,
            {"nll": 0.5},
            {"datasets": [{"train_seeds": [0], "holdout_seeds": [1]}]},
            source_identity(),
        ),
        artifact,
    )
    actor, orientation = module._load_member(artifact, None)
    assert isinstance(actor, FarmActor)
    assert orientation == Orientation.IDENTITY
