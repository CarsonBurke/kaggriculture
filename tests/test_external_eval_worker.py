from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from kaggriculture.opponents import normalize_opponent


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
    agent = module._agent_for(object())
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

    def fake_act_batch(actor, observations, *, deterministic):
        calls.append(actor)
        return type("Out", (), {"actions": [{"actor": actor}]})()

    module.act_batch = fake_act_batch
    agents = [module._agent_for(name) for name in ("first", "second", "third")]
    results = [agent({}) for agent in agents]

    assert [result["actor"] for result in results] == ["first", "second", "third"]
    assert calls == ["first", "second", "third"]


def test_an_exported_artifact_and_a_league_snapshot_both_load_without_a_hint(tmp_path) -> None:
    """The A/B compares a BC artifact against a trained snapshot.

    Routing on the payload's own closed key set is what lets one flag accept
    both; routing on the filename or the caller's promise would not.
    """
    import torch

    from kaggriculture.league import save_actor_snapshot
    from kaggriculture.model import FarmActor, ModelConfig
    from kaggriculture.provenance import source_identity

    module = _script()
    config = ModelConfig(cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3)
    actor = FarmActor(config)

    snapshot = save_actor_snapshot(tmp_path / "league", actor, 3)
    assert isinstance(module._load_member(snapshot.path, None), FarmActor)

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
            "entity-cnn", actor, config, {"nll": 0.5}, {"datasets": []}, source_identity()
        ),
        artifact,
    )
    assert isinstance(module._load_member(artifact, None), FarmActor)
