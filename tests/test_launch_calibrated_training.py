from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

import pytest

from kaggriculture.ppo import PpoConfig, _validate_config
from kaggriculture.production import (
    PRODUCTION_EXTERNAL_EVAL_EVERY,
    PRODUCTION_EXTERNAL_EVAL_OPPONENTS,
    PRODUCTION_ROLLOUT_FORWARD_MODE,
    PRODUCTION_UPDATE_COMPILE_MODE,
    build_training_command,
    production_ppo_config,
    resolve_resume_checkpoint,
)
from kaggriculture.registry import CONV_ENTITY


def test_the_shipped_schedule_is_one_the_update_will_accept() -> None:
    # Every other test builds its own PpoConfig, so nothing until now has asked
    # whether the constants actually launched pass the update's own validation.
    # They are reachable only through a calibration chain and a multi-hour run,
    # which is the most expensive possible place to discover an unrunnable pair.
    shipped = production_ppo_config(update_compile_mode=PRODUCTION_UPDATE_COMPILE_MODE)
    fields = {field.name for field in dataclasses.fields(PpoConfig) if field.init}
    # A field added to PpoConfig and not to the shipped schedule would otherwise
    # be silently defaulted here rather than decided, so the keys are pinned too.
    assert set(shipped) == fields
    _validate_config(PpoConfig(**shipped))

    # The pairing this guards: critic epochs are the critic-only refits that run
    # after the actor's epochs, so a schedule asking for fewer of them than actor
    # epochs describes a loop that cannot run.
    with pytest.raises(ValueError, match="critic epochs"):
        _validate_config(PpoConfig(**{**shipped, "critic_epochs": 0}))


def test_resume_detection_accepts_only_a_regular_latest_checkpoint(tmp_path: Path) -> None:
    assert resolve_resume_checkpoint(tmp_path) is None

    latest = tmp_path / "latest.pt"
    latest.write_bytes(b"atomic checkpoint")
    assert resolve_resume_checkpoint(tmp_path) == latest

    latest.unlink()
    latest.symlink_to(tmp_path / "checkpoint-000005.pt")
    with pytest.raises(ValueError, match="regular file"):
        resolve_resume_checkpoint(tmp_path)

    latest.unlink()
    latest.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        resolve_resume_checkpoint(tmp_path)


def test_training_command_resumes_the_latest_atomic_checkpoint(tmp_path: Path) -> None:
    latest = tmp_path / "run" / "latest.pt"
    command = build_training_command(
        latest.parent,
        iterations=500,
        max_hours=0.0,
        seed=7,
        rollout_forward_mode="eager",
        update_compile_mode="eager",
        expected_source_digest="a" * 64,
        calibration_decision=tmp_path / "decision.json",
        resume_checkpoint=latest,
    )

    assert command[-2:] == ["--resume", str(latest)]
    # External probes are part of the production record: cadence and opponents
    # must both be emitted so the launch command is complete.
    assert command[command.index("--external-eval-every") + 1] == "10"
    assert (
        command[command.index("--external-eval-opponents") + 1]
        == PRODUCTION_EXTERNAL_EVAL_OPPONENTS
    )


def test_training_command_round_trips_through_the_training_parser(monkeypatch, tmp_path) -> None:
    """The command is bound verbatim into run provenance, so its flag list is
    load-bearing: it must parse, validate, and build the production model."""
    from kaggriculture.model import ModelConfig
    from kaggriculture.modelargs import model_config_from_args
    from kaggriculture.registry import CONV_ENTITY, resolve_architecture

    training = _script("train_ppo.py")
    command = build_training_command(
        tmp_path / "run",
        iterations=500,
        max_hours=0.0,
        seed=7,
        rollout_forward_mode=PRODUCTION_ROLLOUT_FORWARD_MODE,
        update_compile_mode=PRODUCTION_UPDATE_COMPILE_MODE,
    )
    monkeypatch.setattr(sys, "argv", ["train_ppo.py", *command[2:]])

    args = training.parse_args()
    training._validate_args(args)

    assert args.architecture == CONV_ENTITY
    assert model_config_from_args(resolve_architecture(args.architecture), args) == ModelConfig()
    # The collection backend and precision move the sampled behavior policy the
    # parity gate bounds, and the update mode moves the graphs that consume it,
    # so the command must state all three rather than inherit a default. Each
    # production constant also has to name a value train_ppo accepts -- argparse
    # choices come from ROLLOUT_FORWARD_MODES and UPDATE_COMPILE_MODES, and this
    # round trip is what enforces that.
    assert command[command.index("--rollout-forward-mode") + 1] == PRODUCTION_ROLLOUT_FORWARD_MODE
    assert "--rollout-bfloat16" in command
    assert command[command.index("--update-compile-mode") + 1] == PRODUCTION_UPDATE_COMPILE_MODE
    assert args.rollout_forward_mode == PRODUCTION_ROLLOUT_FORWARD_MODE
    assert args.rollout_bfloat16 is True
    assert args.update_compile_mode == PRODUCTION_UPDATE_COMPILE_MODE
    assert "--compile-update" not in command


def test_a_population_command_round_trips_and_names_no_opponent_it_never_meets(
    monkeypatch, tmp_path
) -> None:
    """A population wave has no frozen lanes, no built-in lanes and no single
    snapshot to probe, so a command left at the single-learner league values would
    record opponents the run never plays -- and train_ppo would refuse to start.
    One `--init-actor-from` per member, in agent order, is what distinguishes them.
    """
    training = _script("train_ppo.py")
    population = 4
    artifacts = [tmp_path / f"member-{agent}.pt" for agent in range(population)]
    for artifact in artifacts:
        artifact.write_bytes(b"")
    command = build_training_command(
        tmp_path / "run",
        iterations=500,
        max_hours=0.0,
        seed=7,
        rollout_forward_mode=PRODUCTION_ROLLOUT_FORWARD_MODE,
        update_compile_mode=PRODUCTION_UPDATE_COMPILE_MODE,
        population=population,
        games=156,
        initial_actors=artifacts,
        critic_warmup_iterations=10,
    )
    monkeypatch.setattr(sys, "argv", ["train_ppo.py", *command[2:]])

    args = training.parse_args()
    training._validate_args(args)

    assert args.population == population
    assert args.init_actor_from == artifacts
    assert args.league_games == 0
    assert args.league_builtin_lanes == 0
    # A population has no built-in lane. External evaluation stays diagnostic
    # and never enters the training wave; one worker reads each durable
    # checkpoint, so the cadence must land on one.
    assert args.external_eval_every == PRODUCTION_EXTERNAL_EVAL_EVERY
    assert args.external_eval_every % args.checkpoint_every == 0
    # Cost parity at N = 4: 156 games is a multiple of the 12 ordered pairings, so
    # every pairing gets 13 games and seat bias cancels exactly.
    assert args.games % (population * (population - 1)) == 0

    # The wave size is the one thing a launch has to state, since nothing in the
    # tree yet decides between the plan's cost-parity and data-parity candidates.
    with pytest.raises(ValueError, match="multiple of 12"):
        build_training_command(
            tmp_path / "run",
            iterations=500,
            max_hours=0.0,
            seed=7,
            rollout_forward_mode=PRODUCTION_ROLLOUT_FORWARD_MODE,
            update_compile_mode=PRODUCTION_UPDATE_COMPILE_MODE,
            population=population,
        )
    with pytest.raises(ValueError, match="one initial actor per member"):
        build_training_command(
            tmp_path / "run",
            iterations=500,
            max_hours=0.0,
            seed=7,
            rollout_forward_mode=PRODUCTION_ROLLOUT_FORWARD_MODE,
            update_compile_mode=PRODUCTION_UPDATE_COMPILE_MODE,
            population=population,
            games=156,
            initial_actors=artifacts[:2],
        )


def test_warm_started_command_round_trips_through_the_training_parser(monkeypatch, tmp_path):
    """A BC-warm-started baseline has to reach train_ppo through the same
    launcher every family uses, or the families are not being compared on one
    pipeline. The flags must survive the round trip and bind the artifact."""
    training = _script("train_ppo.py")
    artifact = tmp_path / "bc-actor.pt"
    artifact.write_bytes(b"")
    command = build_training_command(
        tmp_path / "run",
        iterations=500,
        max_hours=0.0,
        seed=7,
        rollout_forward_mode="eager",
        update_compile_mode="eager",
        initial_actors=(artifact,),
        critic_warmup_iterations=15,
    )
    monkeypatch.setattr(sys, "argv", ["train_ppo.py", *command[2:]])

    args = training.parse_args()
    training._validate_args(args)

    assert args.init_actor_from == [artifact]
    assert args.critic_warmup_iterations == 15
    assert args.resume is None


def test_warm_start_and_resume_are_rejected_together(tmp_path: Path) -> None:
    """train_ppo rejects the pair; catching it in the builder keeps the
    launcher from rewriting a run's evidence before the run refuses to start."""
    with pytest.raises(ValueError, match="already has an actor"):
        build_training_command(
            tmp_path / "run",
            iterations=500,
            max_hours=0.0,
            seed=7,
            rollout_forward_mode="eager",
            update_compile_mode="eager",
            initial_actors=(tmp_path / "bc-actor.pt",),
            resume_checkpoint=tmp_path / "checkpoint-000010.pt",
        )


def test_critic_warmup_without_a_warm_start_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only to a warm-started run"):
        build_training_command(
            tmp_path / "run",
            iterations=500,
            max_hours=0.0,
            seed=7,
            rollout_forward_mode="eager",
            update_compile_mode="eager",
            critic_warmup_iterations=15,
        )


def test_training_command_requires_digest_and_decision_together(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="together"):
        build_training_command(
            tmp_path,
            iterations=1,
            max_hours=0.0,
            seed=7,
            rollout_forward_mode="eager",
            update_compile_mode="default",
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
    # Each report declares its own two knobs, and the chain's shape is read
    # back out of those declarations, so a fixture has to be able to set them
    # independently -- a single `compiled` flag can only express the two ends
    # of the chain and not the middle report that makes it attributable. Both
    # knobs are mode names rather than flags, so this takes the modes the report
    # declares, which is also what the launcher reads back.
    rollout_forward_mode: str,
    update_compile_mode: str,
    seconds: float,
    # The two phases are timed independently and decided independently, so
    # the builder has to be able to move them independently. The default
    # keeps them proportional, which is the case where a per-phase decision
    # and a blended one agree.
    rollout_share: float = 0.7,
    # Opponent reconstruction is a real term of the iteration total that
    # neither knob moves, so it rides on top of the two phases rather than
    # inside either. `seconds` stays the compute total the phase shares divide,
    # which keeps every ratio in these tests exact: a constant factor on both
    # sides of a speedup cancels.
    setup_share: float = 0.01,
    seed: int = 20260812,
    source_digest: str | None = None,
    game_counts: list[int] | None = None,
) -> list[dict[str, object]]:
    game_counts = [64, 112, 128] if game_counts is None else game_counts
    # Enough repeats that the steady set is a real sample rather than the one
    # iteration a two-repeat run leaves after the cold start is dropped. The
    # per-iteration seconds vary across the steady set so the medians here are
    # medians, not a single value wearing the name.
    repeats = module.MINIMUM_CALIBRATION_REPEATS
    configuration: dict[str, object] = {
        "event": "configuration",
        # Every key the benchmark emits belongs here, knob or not: the launcher
        # compares all of them across the chain, so a key this fixture omits is
        # a key nothing in this file exercises. The test at the end of this
        # module is what keeps the two in step.
        "architecture": CONV_ENTITY,
        "rollout_forward_mode": rollout_forward_mode,
        # Not a knob: fixed on every node, and the launcher pins it to
        # production's value, so a chain that moved it is rejected rather than
        # attributed to whichever knob its step named.
        "rollout_bfloat16": module.PRODUCTION_ROLLOUT_BFLOAT16,
        "device": "cuda",
        "hardware": _hardware(),
        "self_play_game_counts": game_counts,
        "league_games_per_iteration": 96,
        "league_opponents": 4,
        "league_active_opponents": 2,
        "league_historical_opponents": 2,
        "episode_steps": 720,
        "physical_games_per_iteration": [games + 96 for games in game_counts],
        "repeats": repeats,
        "seed": seed,
        "temperature": 1.0,
        "precision": {
            "use_bfloat16": True,
            "float32_matmul_precision": "high",
            "cudnn_benchmark": True,
        },
        "model": module.production_model_config(),
        "ppo": module.production_ppo_config(update_compile_mode=update_compile_mode),
        "max_update_replay_kl": module.MAX_UPDATE_REPLAY_KL,
        "max_update_replay_tail_fraction": module.MAX_UPDATE_REPLAY_TAIL_FRACTION,
        "max_first_minibatch_kl": module.MAX_FIRST_MINIBATCH_KL,
        "max_value_target_saturated_fraction": module.MAX_VALUE_TARGET_SATURATED_FRACTION,
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
            # Spread the steady iterations symmetrically about the intended
            # median so the medians below are computed from a set that actually
            # varies, while staying exactly predictable. A fixture whose steady
            # values are all identical cannot tell a median apart from a point
            # measurement, which is the thing this report format exists to
            # distinguish.
            # Steady repeats are 1..repeats-1, so their median index is
            # repeats/2 -- exact while repeats is even, which the constant is.
            jitter = 1.0 + (repeat - repeats / 2.0) * 0.01
            compute_seconds = steady_seconds * 1.2 if repeat == 0 else steady_seconds * jitter
            rollout_seconds = compute_seconds * rollout_share
            update_seconds = compute_seconds - rollout_seconds
            setup_seconds = compute_seconds * setup_share
            total_seconds = setup_seconds + rollout_seconds + update_seconds
            record: dict[str, object] = {
                "event": "iteration",
                "phase": "cold_start" if repeat == 0 else "steady_state",
                "repeat": repeat,
                "self_play_games": games,
                "league_games": 96,
                "physical_games": games + 96,
                "opponent_setup_seconds": setup_seconds,
                "rollout_seconds": rollout_seconds,
                "update_replay_parity_seconds": compute_seconds * 0.05,
                "update_seconds": update_seconds,
                "total_seconds": total_seconds,
                "iterations_per_hour": 3600.0 / total_seconds,
                "physical_games_per_rollout_second": (games + 96) / rollout_seconds,
                "critic_replayed_states_per_second": 1000.0 / update_seconds,
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
                **{
                    f"steady_{key}_median": statistics.median(row[key] for row in iterations[1:])
                    for key in (
                        "total_seconds",
                        "opponent_setup_seconds",
                        "rollout_seconds",
                        "update_seconds",
                        "iterations_per_hour",
                        "physical_games_per_rollout_second",
                        "critic_replayed_states_per_second",
                    )
                },
            }
        )
    records.append(module._expected_completion(game_counts, repeats))
    return records


#: The order the chain turns its knobs on. The update knob goes first because
#: that is the order the archive prescribes: it is the knob with a large
#: measured win, so the report that isolates the rollout knob is the one that
#: already has the update compiled -- which is also the configuration a run
#: with the update compiled would actually be launched in.
_CHAIN_STEPS = ("update_compile_mode", "rollout_forward_mode")
#: What each knob declares while its phase is uncompiled and once its step has
#: turned it on. Both knobs' values are mode names -- `inductor` on the
#: collection side, being the mode measurement selects, and `default` on the
#: update side, Inductor fusion without graph capture. A fixture writing `True`
#: for either would describe a report the launcher refuses, since the mode is
#: what the phase actually consults.
_UNCOMPILED_KNOBS = {"rollout_forward_mode": "eager", "update_compile_mode": "eager"}
_COMPILED_KNOBS = {"rollout_forward_mode": "inductor", "update_compile_mode": "default"}


def _chain(
    module,
    *,
    rollout: tuple[float, float, float],
    update: tuple[float, float, float],
    # Opponent reconstruction belongs to no knob, so it is held across the
    # chain unless a test is specifically about a step drifting it.
    setup: tuple[float, float, float] = (0.1, 0.1, 0.1),
    steps: tuple[str, str] = _CHAIN_STEPS,
    **records_kwargs: object,
) -> list[list[dict[str, object]]]:
    """A valid three-node chain from all-eager to all-compiled.

    `rollout`, `update` and `setup` state each node's intended steady phase
    median directly, in chain order, because that is the physics the decision
    tests are about: a step moves one phase and holds the others, so a node's
    phases are not a scaled copy of the previous node's. A blended total and a
    share cannot express a chain in which one phase halves while the other
    stands still, which is exactly the case a per-knob attribution exists for.

    Every node is still built by `_records`, so the cold start is still dropped
    and the steady iterations still vary about the intended value: each median
    below is a median of a real sample rather than a point measurement wearing
    the name.
    """
    knobs = [dict(_UNCOMPILED_KNOBS), dict(_UNCOMPILED_KNOBS), dict(_COMPILED_KNOBS)]
    knobs[1][steps[0]] = _COMPILED_KNOBS[steps[0]]
    return [
        _records(
            module,
            rollout_forward_mode=node["rollout_forward_mode"],
            update_compile_mode=node["update_compile_mode"],
            seconds=rollout[index] + update[index],
            rollout_share=rollout[index] / (rollout[index] + update[index]),
            setup_share=setup[index] / (rollout[index] + update[index]),
            **records_kwargs,
        )
        for index, node in enumerate(knobs)
    ]


def _set_steady_phases(
    records: list[dict[str, object]],
    phases: list[tuple[float, float, float]],
    games: int = 112,
) -> list[dict[str, object]]:
    """Rewrite one batch's steady iterations to the given phase seconds.

    `_records` moves the three phases proportionally, which is the realistic
    case and also the one where a sum of phase medians and a median of totals
    happen to coincide. Tests about that distinction need a report whose steady
    iterations disagree about which of them was the slow one. Every rewritten
    row still satisfies the per-iteration identity and every derived rate, so
    nothing the report format itself guarantees is being faked away.

    `phases` is one `(setup, rollout, update)` triple per steady iteration, in
    order, and the batch summary is re-derived from them.
    """
    rewritten = [dict(record) for record in records]
    rows = [
        record
        for record in rewritten
        if record.get("event") == "iteration" and record.get("self_play_games") == games
    ]
    # The cold start is dropped from every median, so it is left alone.
    steady = rows[1:]
    if len(steady) != len(phases):
        raise AssertionError(f"{len(steady)} steady iterations, {len(phases)} phase triples")
    for record, (setup, rollout, update) in zip(steady, phases, strict=True):
        record["opponent_setup_seconds"] = setup
        record["rollout_seconds"] = rollout
        record["update_seconds"] = update
        record["total_seconds"] = setup + rollout + update
        record["iterations_per_hour"] = 3600.0 / record["total_seconds"]
        record["physical_games_per_rollout_second"] = (games + 96) / rollout
        record["critic_replayed_states_per_second"] = 1000.0 / update
    summary = next(
        record
        for record in rewritten
        if record.get("event") == "batch_summary" and record.get("self_play_games") == games
    )
    for key in (
        "total_seconds",
        "opponent_setup_seconds",
        "rollout_seconds",
        "update_seconds",
        "iterations_per_hour",
        "physical_games_per_rollout_second",
        "critic_replayed_states_per_second",
    ):
        summary[f"steady_{key}_median"] = statistics.median(row[key] for row in steady)
    return rewritten


def _decide_with_divergence_allowed(module, chain: list[list[dict[str, object]]]):
    """`choose_compilation` with the budget-consistency bound lifted.

    The bound and the attribution arithmetic are two independent defences
    against the same misreading, and a test that only sees the first cannot
    tell whether the second exists. Lifting it lets the arithmetic be checked
    on a chain the bound would otherwise reject first.
    """
    saved = module.MAXIMUM_PHASE_BUDGET_DIVERGENCE
    module.MAXIMUM_PHASE_BUDGET_DIVERGENCE = math.inf
    try:
        return module.choose_compilation(chain)
    finally:
        module.MAXIMUM_PHASE_BUDGET_DIVERGENCE = saved


def _write_report(path: Path, records: list[dict[str, object]]) -> bytes:
    contents = ("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n").encode()
    path.write_bytes(contents)
    return contents


def _production_summary(records: list[dict[str, object]]) -> dict[str, object]:
    """The batch summary the decision is read from, at the production batch."""
    return next(
        record
        for record in records
        if record.get("event") == "batch_summary" and record.get("self_play_games") == 112
    )


#: A chain in which both knobs are worth having: the update knob halves a 20 s
#: update, then the rollout knob halves a 10 s rollout, over a 0.1 s setup that
#: neither touches. Node totals are 30.1 s, 20.1 s and 15.1 s. Shared by the
#: tests that are about something other than the timings, so it is visible at a
#: glance which numbers a test actually depends on.
_EARNED = {"rollout": (10.0, 10.0, 5.0), "update": (20.0, 10.0, 10.0)}


def test_a_chain_whose_every_step_earns_its_keep_compiles_every_knob() -> None:
    """The decided configuration is a node of the chain, so its total is measured.

    Knobs are enabled as a prefix of the chain precisely so that the
    configuration which will run is one that was timed. The decision this
    replaces projected a total by taking the faster side of each phase, which
    describes an iteration no report contains; the decided total is now the
    decided node's own steady median, to the bit.
    """
    module = _script()
    chain = _chain(module, **_EARNED)

    decision = module.choose_compilation(chain)

    assert decision["rollout_forward_mode"] == "inductor"
    assert decision["update_compile_mode"] == "default"
    assert decision["chain"] == ["eager", "mixed", "compiled"]
    assert decision["chain_steps"] == ["update_compile_mode", "rollout_forward_mode"]
    assert decision["minimum_compile_speedup"] == module.MINIMUM_COMPILE_SPEEDUP
    assert decision["measured_compile_speedup"] == pytest.approx(30.1 / 15.1)
    assert decision["attributed_knob_speedups"] == {
        "update_compile_mode": pytest.approx(30.1 / 20.1),
        "rollout_forward_mode": pytest.approx(20.1 / 15.1),
    }
    assert decision["knob_evidence"]["update_compile_mode"]["isolated_by"] == ["eager", "mixed"]
    assert decision["decided_configuration"] == "compiled"
    assert (
        decision["decided_steady_total_seconds"]
        == _production_summary(chain[-1])["steady_total_seconds_median"]
    )
    assert decision["eager_steady_total_seconds"] == pytest.approx(30.1)
    assert decision["compiled_steady_total_seconds"] == pytest.approx(15.1)
    assert decision["self_play_games"] == 112
    assert decision["league_games"] == 96
    assert decision["validated_evidence"]["eager"][-1]["completed"] is True
    # Both removed keys described a configuration no report measured: the
    # projection interpolated one, and the per-phase ratios it was built from
    # were differenced across a pair that moved both knobs at once.
    assert "projected_steady_total_seconds" not in decision
    assert "measured_phase_speedups" not in decision


def test_a_chain_whose_steps_are_both_marginal_compiles_nothing() -> None:
    """Compilation is not free, so a couple of percent does not buy it.

    It costs warmup, replay divergence against the collector, and a decision
    stamped into run provenance that cannot be revised mid-run. Each step here
    moves its own phase by 2%, which is under the floor once spread over the
    whole iteration, and the pair of them together is still under it.
    """
    module = _script()

    decision = module.choose_compilation(
        _chain(module, rollout=(10.0, 10.0, 9.8), update=(20.0, 19.6, 19.6))
    )

    # A knob a chain leaves off reads as the mode that compiles nothing, not as
    # a `False` the phase could not act on.
    assert decision["rollout_forward_mode"] == "eager"
    assert decision["update_compile_mode"] == "eager"
    assert decision["decided_configuration"] == "eager"
    assert decision["decided_steady_total_seconds"] == pytest.approx(30.1)
    assert decision["measured_compile_speedup"] == pytest.approx(30.1 / 29.5)
    assert all(
        speedup < module.MINIMUM_COMPILE_SPEEDUP
        for speedup in decision["attributed_knob_speedups"].values()
    )


def test_a_knob_is_decided_against_the_report_that_isolates_it() -> None:
    """A knob compilation loses must not ride along with a knob it wins.

    This is the arrangement the conv model measured: compiling the update saved
    a great deal, compiling the rollout collector moved only the rollout phase
    and only by 2.7%, and the all-eager to all-compiled total still cleared the
    threshold by a mile. A decision that reads only that total enables the
    losing knob on the winning knob's evidence, which is the one thing a
    calibration exists to stop -- and it is the middle report that lets the
    rollout knob be measured at all, since the outer pair moves both at once.
    """
    module = _script()

    decision = module.choose_compilation(
        _chain(module, rollout=(8.5054, 8.5054, 8.2818), update=(28.2, 8.4, 8.4))
    )

    # The end-to-end total would have compiled both, which is what makes this
    # the case worth a test rather than an academic one.
    assert decision["measured_compile_speedup"] == pytest.approx(36.8054 / 16.7818)
    assert decision["measured_compile_speedup"] >= module.MINIMUM_COMPILE_SPEEDUP
    assert decision["update_compile_mode"] == "default"
    assert decision["rollout_forward_mode"] == "eager"
    assert decision["attributed_knob_speedups"]["update_compile_mode"] == pytest.approx(
        36.8054 / 17.0054
    )
    assert decision["attributed_knob_speedups"]["rollout_forward_mode"] == pytest.approx(
        17.0054 / 16.7818
    )
    assert decision["attributed_knob_speedups"]["rollout_forward_mode"] < 1.05
    assert decision["decided_configuration"] == "mixed"
    # The decided node is the mixed one, so the total the launcher records is
    # that node's own measured median rather than a configuration nobody ran.
    assert decision["decided_steady_total_seconds"] == pytest.approx(17.0054)
    evidence = decision["knob_evidence"]["rollout_forward_mode"]
    # The rollout knob is scored against the node that already has the update
    # compiled, not against all-eager: that is the configuration it would run
    # in, and it is the pair in which nothing but the rollout moved.
    assert evidence["isolated_by"] == ["mixed", "compiled"]
    assert evidence["phase_speedup"] == pytest.approx(8.5054 / 8.2818)
    assert evidence["held_phase_drift"] == {
        "steady_opponent_setup_seconds_median": pytest.approx(1.0),
        "steady_update_seconds_median": pytest.approx(1.0),
    }


def test_a_knobs_attribution_ignores_drift_in_the_phases_its_step_held() -> None:
    """Holding every other phase at its measured value is what buys the attribution.

    The conv calibration this design replaces credited the rollout knob with
    8.0% off a pair whose rollout phase differed by 4.9% with the knob
    unchanged. Swapping only the knob's own phase inside the isolating node's
    iteration budget makes the number immune to that: the step's raw total
    ratio moves with the drift, the attribution does not, and the drift is
    reported beside it so a reader can see how much of the pair was noise.
    """
    module = _script()
    steady = module.choose_compilation(_chain(module, **_EARNED))
    # Same rollout halving, but the mixed to compiled step also lets the update
    # phase slip 6% and the compile-invariant setup term slip 10%.
    drifted = module.choose_compilation(
        _chain(
            module,
            rollout=(10.0, 10.0, 5.0),
            update=(20.0, 10.0, 10.6),
            setup=(0.1, 0.1, 0.11),
        )
    )

    steady_evidence = steady["knob_evidence"]["rollout_forward_mode"]
    drifted_evidence = drifted["knob_evidence"]["rollout_forward_mode"]
    assert drifted_evidence["attributed_iteration_speedup"] == pytest.approx(
        steady_evidence["attributed_iteration_speedup"]
    )
    assert drifted_evidence["step_total_speedup"] == pytest.approx(20.1 / 15.71)
    assert drifted_evidence["step_total_speedup"] != pytest.approx(
        steady_evidence["step_total_speedup"]
    )
    assert drifted_evidence["held_phase_drift"] == {
        "steady_opponent_setup_seconds_median": pytest.approx(1.1),
        "steady_update_seconds_median": pytest.approx(1.06),
    }
    assert steady_evidence["held_phase_drift"] == {
        "steady_opponent_setup_seconds_median": pytest.approx(1.0),
        "steady_update_seconds_median": pytest.approx(1.0),
    }
    # The drift is not large enough to change the outcome here, which is the
    # point: the decision is the same because the attribution did not move.
    assert drifted["rollout_forward_mode"] == "inductor"
    assert drifted["decided_configuration"] == "compiled"


def test_a_large_speedup_on_a_small_phase_does_not_clear_the_iteration_floor() -> None:
    """The floor is on the iteration, so a knob is worth what it saves per iteration.

    A per-phase ratio flatters a knob whose phase is a small share of the run:
    five times faster on a half-second phase of a ten-second iteration is under
    4% of the iteration, which is not worth a warmup cost and an irreversible
    provenance stamp. The rejection has to come from the whole-iteration number
    while the phase ratio is left visible beside it, or nobody reading the
    decision can tell a small phase from a knob that did nothing.
    """
    module = _script()

    decision = module.choose_compilation(
        _chain(module, rollout=(0.5, 0.5, 0.1), update=(40.0, 10.0, 10.0))
    )

    evidence = decision["knob_evidence"]["rollout_forward_mode"]
    assert evidence["phase_speedup"] == pytest.approx(5.0)
    assert evidence["attributed_iteration_speedup"] == pytest.approx(10.6 / 10.2)
    assert evidence["attributed_iteration_speedup"] < module.MINIMUM_COMPILE_SPEEDUP
    assert decision["rollout_forward_mode"] == "eager"
    assert decision["update_compile_mode"] == "default"
    assert decision["decided_configuration"] == "mixed"


def test_a_knob_that_only_clears_the_floor_behind_a_failing_knob_is_refused() -> None:
    """A late knob was measured on top of a configuration this run will not use.

    Knobs are enabled as a prefix of the chain, so if the first step misses the
    floor the second step's evidence describes a node that would never be
    launched -- the chain says nothing about what that knob is worth on its
    own. Guessing either way is wrong: enabling it credits it with a pairing it
    was never measured in, and dropping it silently discards a real win. The
    chain that answers the question is one benchmark away.
    """
    module = _script()

    with pytest.raises(ValueError, match=r"only on top of update_compile_mode"):
        module.choose_compilation(
            _chain(module, rollout=(20.0, 20.0, 10.0), update=(10.0, 9.8, 9.8))
        )


def test_a_chain_that_is_not_a_single_knob_walk_from_all_eager_is_rejected() -> None:
    """Every attribution reads one step as one knob, so the shape is load-bearing.

    The shape is taken from what the reports declare rather than from an
    argument, which is what stops a mislabelled pair from naming a knob neither
    run moved. These are the four ways a supplied chain can fail to isolate
    anything, and each has to be refused rather than scored.
    """
    module = _script()
    chain = _chain(module, **_EARNED)

    # Two reports can only be differenced across both knobs at once.
    with pytest.raises(ValueError, match="needs 3 reports"):
        module.choose_compilation(chain[:2])

    # A chain starting with a knob already on never measures that knob: no pair
    # in it brackets the knob's own change. The value is reported beside the knob
    # because "on" is a particular mode now, and which one it was is what a
    # reader has to see to fix the chain.
    with pytest.raises(
        ValueError,
        match=r"must start all-eager; its first report has \[.update_compile_mode=.default..\]",
    ):
        module.choose_compilation([chain[1], chain[1], chain[2]])

    # The same on the other knob, which the chain turns second, so the fixture
    # has to reorder the steps to put it first.
    mode_first = _chain(module, **_EARNED, steps=("rollout_forward_mode", "update_compile_mode"))
    with pytest.raises(
        ValueError,
        match=r"must start all-eager; its first report has "
        r"\[.rollout_forward_mode=.inductor..\]",
    ):
        module.choose_compilation([mode_first[1], mode_first[1], mode_first[2]])

    # A step that moves both knobs is the contaminated pair the middle report
    # exists to replace.
    with pytest.raises(ValueError, match="must turn exactly one knob on"):
        module.choose_compilation([chain[0], chain[2], chain[2]])

    # And a step that turns a knob back off walks away from all-compiled, so
    # the chain has no node in which both knobs are on to compare against.
    with pytest.raises(ValueError, match="must turn exactly one knob on"):
        module.choose_compilation([chain[0], chain[1], chain[0]])

    # A step from one compiling mode to another moves the knob without leaving
    # its uncompiled value, so the pair called on to isolate it is two compiled
    # runs. Substituting one node's phase median into the other's total then
    # answers "cudagraphs against inductor", which is not the question the floor
    # is applied to, and the chain has no measurement of compiling at all.
    walk = [
        _records(module, rollout_forward_mode=mode, update_compile_mode="eager", seconds=10.0)
        for mode in ("eager", "cudagraphs", "inductor")
    ]
    with pytest.raises(ValueError, match="must turn exactly one knob on"):
        module.choose_compilation(walk)

    # The same walk on the update side, which the boolean could not even
    # express: `default` to `max-autotune` is a knob that moved without ever
    # having been off.
    update_walk = [
        _records(module, rollout_forward_mode="eager", update_compile_mode=mode, seconds=10.0)
        for mode in ("eager", "default", "max-autotune")
    ]
    with pytest.raises(ValueError, match="must turn exactly one knob on"):
        module.choose_compilation(update_walk)

    # A mode outside its domain dies here, at the last point that can see the
    # domain: `provenance` re-derives this decision inside a submission bundle
    # that ships neither `rollout.py` nor `ppo.py`, so it can only check that
    # each knob is mode-shaped. The two domains are disjoint, so each knob
    # wearing the other's mode is the case that has to be caught.
    with pytest.raises(ValueError, match="does not declare a collection forward mode"):
        module.choose_compilation(
            [
                _records(
                    module,
                    rollout_forward_mode="reduce-overhead",
                    update_compile_mode="eager",
                    seconds=10.0,
                ),
                chain[1],
                chain[2],
            ]
        )
    with pytest.raises(ValueError, match="does not declare an update compile mode"):
        module.choose_compilation(
            [
                _records(
                    module,
                    rollout_forward_mode="eager",
                    update_compile_mode="cudagraphs",
                    seconds=10.0,
                ),
                chain[1],
                chain[2],
            ]
        )


def test_an_iteration_whose_phases_do_not_sum_to_its_total_is_rejected() -> None:
    """Every summary median is checked against the median of the same key, so a
    report whose phases and total disagree is internally consistent at the
    summary level and passes every other gate.

    That matters because each knob's attribution substitutes one phase median
    into an iteration total. Left unchecked, a report claiming one-second
    phases beside a hundred-second total would make that substitution
    arithmetic on unrelated numbers.
    """
    module = _script()

    for key in ("opponent_setup_seconds", "rollout_seconds", "update_seconds"):
        chain = _chain(module, **_EARNED)
        iterations = [row for row in chain[0] if row.get("event") == "iteration"]
        # Halve one phase and leave the total and every summary median alone.
        iterations[1][key] = iterations[1][key] / 2.0
        with pytest.raises(ValueError, match="is not its rollout and update phases summed"):
            module.choose_compilation(chain)

    # The derived rate has to describe the total it came from, too, or an
    # honest total can carry an inflated throughput beside it.
    chain = _chain(module, **_EARNED)
    iterations = [row for row in chain[1] if row.get("event") == "iteration"]
    iterations[1]["iterations_per_hour"] = iterations[1]["iterations_per_hour"] * 2.0
    with pytest.raises(ValueError, match="iterations per hour"):
        module.choose_compilation(chain)


def test_a_summary_whose_phase_medians_describe_no_iteration_is_rejected() -> None:
    """A median of sums is not a sum of medians, and the gap is not bounded.

    Every iteration is checked to satisfy total = setup + rollout + update
    exactly, and every summary median is checked against the median of the same
    key. Both hold here. What they do not constrain is the relationship between
    the three phase medians and the total median: when the steady iterations
    disagree about which of them was slow, the medians come from different rows
    and their sum describes no iteration in the set.

    The chain below is the shape that exploits it. Rollout is bimodal and
    identical on every node; the update phase is arranged so that its median
    lands on the rows where rollout is cheap. Each node's phase medians sum to
    about 2 s beside a measured 10 s iteration, so an attribution taken on that
    sum reads a knob worth 1.05 whose step made the iteration 1% faster.
    """
    module = _script()

    rollout = [1.0, 1.0, 1.0, 9.0, 9.0]
    updates = ([9.0, 9.0, 1.0, 1.0, 1.0], [9.0, 9.0, 0.9, 0.9, 0.9], [9.0, 9.0, 0.9, 0.9, 0.9])
    chain = [
        _set_steady_phases(node, [(0.1, r, u) for r, u in zip(rollout, update, strict=True)])
        for node, update in zip(_chain(module, **_EARNED), updates, strict=True)
    ]

    summary = _production_summary(chain[0])
    budget = sum(summary[key] for key in module.PHASE_MEDIANS)
    total = summary["steady_total_seconds_median"]
    assert budget == pytest.approx(2.1)
    assert total == pytest.approx(10.1)

    with pytest.raises(ValueError, match="describe no iteration"):
        module.choose_compilation(chain)

    # And the arithmetic no longer depends on that gate catching it: with the
    # bound relaxed the same chain reads the honest whole-iteration ratio and
    # compiles nothing, because the attribution is anchored on the measured
    # total rather than on the summed medians. The two defences are
    # independent, which is why both are here.
    decision = _decide_with_divergence_allowed(module, chain)
    assert decision["attributed_knob_speedups"]["update_compile_mode"] == pytest.approx(10.1 / 10.0)
    assert decision["update_compile_mode"] == "eager"
    assert decision["rollout_forward_mode"] == "eager"


def test_a_knobs_attribution_is_anchored_on_a_measured_iteration_total() -> None:
    """The counterfactual is about an iteration, so it starts from a real one.

    Summing the three steady phase medians produces a number that is close to
    the measured total and is not it. Building the ratio on that sum instead of
    on the total means neither end of the counterfactual is an iteration the
    benchmark ran, and the error it carries is unrelated to the knob.

    The fixture below keeps the divergence honest and small -- half a percent,
    inside what `MAXIMUM_PHASE_BUDGET_DIVERGENCE` allows and eight times what a
    real matched report measures -- so this pins which of the two numbers the
    decision is built from rather than re-testing the rejection above.
    """
    module = _script()

    # Three of the five steady iterations are the same; the other two are a
    # little slower in one phase each, and no row carries both. So the phase
    # medians all come from the fast rows and the total median does not.
    setup = 0.1
    node = _set_steady_phases(
        _chain(module, **_EARNED)[0],
        [
            (setup, 100.0, 200.0),
            (setup, 100.0, 200.0),
            (setup, 100.0, 201.5),
            (setup, 101.5, 200.0),
            (setup, 102.0, 200.0),
        ],
    )
    summary = _production_summary(node)
    budget = sum(summary[key] for key in module.PHASE_MEDIANS)
    total = summary["steady_total_seconds_median"]
    assert budget == pytest.approx(300.1)
    assert total == pytest.approx(301.6)
    assert abs(budget - total) / total < module.MAXIMUM_PHASE_BUDGET_DIVERGENCE

    chain = [node, *_chain(module, **_EARNED)[1:]]
    decision = module.choose_compilation(chain)

    # The update knob is the first step, so its `before` node is the one above.
    after = _production_summary(chain[1])["steady_update_seconds_median"]
    before = summary["steady_update_seconds_median"]
    attributed = decision["attributed_knob_speedups"]["update_compile_mode"]
    assert attributed == pytest.approx(total / (total - before + after))
    # The fixture discriminates: the retired form disagrees in the third digit,
    # which is a fifth of the distance the floor sits above one.
    on_the_budget = budget / (budget - before + after)
    assert abs(attributed - on_the_budget) > 0.01


def test_compile_decision_rejects_nonproduction_or_mismatched_configuration() -> None:
    module = _script()
    nonproduction_model = _chain(module, **_EARNED)
    nonproduction_model[2][0]["model"] = dict(nonproduction_model[2][0]["model"], cnn_width=128)

    with pytest.raises(ValueError, match="model"):
        module.choose_compilation(nonproduction_model)

    # Two nodes differing in anything but the knobs are not a chain: the step
    # between them changed more than the knob it claims.
    other_gpu = _chain(module, **_EARNED)
    other_gpu[0][0]["hardware"] = dict(other_gpu[0][0]["hardware"], device_name="Other GPU")
    with pytest.raises(ValueError, match=r"configurations differ.*hardware"):
        module.choose_compilation(other_gpu)

    # Agreeing with each other is not enough: every node is also checked
    # against production, or the chain measures some other training run.
    nonproduction_ppo = _chain(module, **_EARNED)
    for records in nonproduction_ppo:
        records[0]["ppo"] = dict(records[0]["ppo"], epochs=4)
    with pytest.raises(ValueError, match=r"ppo.*production"):
        module.choose_compilation(nonproduction_ppo)


def test_a_calibration_may_time_only_the_production_batch_on_every_node() -> None:
    """The decision reads one median, so timing four batch sizes to get it is waste.

    The saving belongs on every node rather than on eager alone: a swept node
    compared against an unswept one compares two protocols as well as two
    configurations, and the archive cannot separate those because every source
    digest was measured under exactly one sweep. Symmetric costs less and
    claims less.
    """
    module = _script()

    decision = module.choose_compilation(_chain(module, **_EARNED, game_counts=[112]))

    assert decision["rollout_forward_mode"] == "inductor"
    assert decision["update_compile_mode"] == "default"
    assert decision["measured_compile_speedup"] == pytest.approx(30.1 / 15.1)

    # Symmetric, and still only accepted for a sweep that contains the batch
    # the decision is read from.
    with pytest.raises(ValueError, match="including 112"):
        module.choose_compilation(_chain(module, **_EARNED, game_counts=[64, 128]))

    # An asymmetric chain is rejected: the sweep is part of what has to match,
    # so a shortened node cannot be compared against a swept one.
    asymmetric = _chain(module, **_EARNED, game_counts=[112])
    asymmetric[2] = _chain(module, **_EARNED)[2]
    with pytest.raises(ValueError, match=r"configurations differ.*self_play_game_counts"):
        module.choose_compilation(asymmetric)


def test_compile_decision_rejects_incomplete_or_fabricated_summary() -> None:
    module = _script()

    truncated = _chain(module, **_EARNED)
    truncated[0] = truncated[0][:-1]
    with pytest.raises(ValueError, match="incomplete"):
        module.choose_compilation(truncated)

    # Every summary median the decision reads must be recomputed from the
    # iteration records, the per-phase ones most of all: those are the terms of
    # the iteration budget each knob is attributed inside, so a summary nobody
    # checks is a knob nobody measured.
    for key in (
        "steady_total_seconds_median",
        "steady_opponent_setup_seconds_median",
        "steady_rollout_seconds_median",
        "steady_update_seconds_median",
    ):
        fabricated = _chain(module, **_EARNED)
        _production_summary(fabricated[1])[key] = 1.0
        with pytest.raises(ValueError, match="does not match iterations"):
            module.choose_compilation(fabricated)

    nonfinite = _chain(module, **_EARNED)
    nonfinite[2][1]["total_seconds"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        module.choose_compilation(nonfinite)


def test_source_digest_is_validated_and_must_match() -> None:
    module = _script()

    with pytest.raises(ValueError, match="source_digest"):
        module.choose_compilation(_chain(module, **_EARNED, source_digest="invalid"))

    # One node measured against a different tree is still the whole chain
    # invalidated, whichever node it is.
    mismatched = _chain(module, **_EARNED)
    mismatched[1][0]["source_digest"] = "b" * 64
    with pytest.raises(ValueError, match="source_digest"):
        module.choose_compilation(mismatched)


def test_report_reader_hashes_exact_bytes_and_rejects_nonstandard_json(tmp_path: Path) -> None:
    module = _script()
    report = tmp_path / "report.jsonl"
    contents = _write_report(
        report,
        _records(module, rollout_forward_mode="eager", update_compile_mode="eager", seconds=10.0),
    )

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
    paths = {name: tmp_path / f"{name}.jsonl" for name in ("eager", "mixed", "compiled")}
    records = dict(zip(paths, _chain(module, **_EARNED), strict=True))
    contents = {name: _write_report(paths[name], records[name]) for name in paths}
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
            str(paths["eager"]),
            # The middle report is what makes each knob attributable, so the
            # launcher requires it rather than accepting the outer pair alone.
            "--mixed-report",
            str(paths["mixed"]),
            "--compiled-report",
            str(paths["compiled"]),
            "--run-dir",
            str(run_directory),
            "--iterations",
            "17",
        ],
    )

    with pytest.raises(Executed):
        module.main()

    decision = json.loads((run_directory / "calibration-decision.json").read_text())
    # Every node of the chain is retained byte for byte and hashed, not just
    # the two ends: the decision cannot be re-derived from the outer pair.
    for name, blob in contents.items():
        assert decision[f"{name}_report_sha256"] == hashlib.sha256(blob).hexdigest()
        assert decision[f"{name}_report_size_bytes"] == len(blob)
        assert Path(decision[f"{name}_report"]).read_bytes() == blob
    assert set(decision["validated_evidence"]) == {"eager", "mixed", "compiled"}
    assert decision["validated_evidence"] == records
    assert invocation["executable"] == sys.executable
    assert invocation["command"] == decision["training_command"]
    assert decision["resume_checkpoint"] == str(latest_checkpoint)
    assert decision["training_command"][-2:] == ["--resume", str(latest_checkpoint)]
    forward_mode = decision["training_command"].index("--rollout-forward-mode") + 1
    assert decision["training_command"][forward_mode] == "inductor"
    assert "--rollout-bfloat16" in decision["training_command"]
    assert "--compile-rollout" not in decision["training_command"]
    assert "--compile-update" not in decision["training_command"]
    update_mode = decision["training_command"].index("--update-compile-mode") + 1
    assert decision["training_command"][update_mode] == "default"
    # Entropy remains measured but never enters the optimized objective.
    assert "--entropy-coefficient" not in decision["training_command"]
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
            str(module.production_ppo_config(update_compile_mode="eager")["actor_gae_lambda"]),
        ),
        (
            "--target-kl",
            str(module.production_ppo_config(update_compile_mode="eager")["target_kl"]),
        ),
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
    # The direct launcher carries no measurement, so it takes the standing
    # per-phase evidence: the update runs the mode production has been running,
    # and collection runs the mode and precision the rollout sweep and the
    # parity gate both selected.
    assert launch["rollout_forward_mode"] == PRODUCTION_ROLLOUT_FORWARD_MODE
    assert launch["rollout_bfloat16"] is True
    assert launch["update_compile_mode"] == PRODUCTION_UPDATE_COMPILE_MODE
    assert launch["iterations"] == 17
    assert launch["resume_checkpoint"] == str(latest_checkpoint)
    assert launch["source_identity"] == module.source_identity()
    assert invocation["executable"] == sys.executable
    assert invocation["command"] == launch["training_command"]
    assert "--compile-rollout" not in launch["training_command"]
    assert (
        launch["training_command"][launch["training_command"].index("--rollout-forward-mode") + 1]
        == PRODUCTION_ROLLOUT_FORWARD_MODE
    )
    assert "--rollout-bfloat16" in launch["training_command"]
    assert "--compile-update" not in launch["training_command"]
    assert (
        launch["training_command"][launch["training_command"].index("--update-compile-mode") + 1]
        == PRODUCTION_UPDATE_COMPILE_MODE
    )
    assert "--expected-source-digest" not in launch["training_command"]
    assert "--calibration-decision" not in launch["training_command"]
    assert launch["training_command"][-2:] == ["--resume", str(latest_checkpoint)]


def test_relaunching_a_warm_started_run_resumes_and_keeps_naming_the_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resubmitting the identical launch command is how a killed run continues.

    The warm start belongs to the run, not to the relaunch: once a checkpoint
    exists the actor and the remaining critic warmup both come from it, which
    is why train_ppo refuses to have them restated. The launcher therefore
    has to drop the flags itself rather than fail, and it has to carry the
    recorded warm start forward -- it rewrites the decision file in place, so
    restating the now-empty flags would leave the run's own launch record
    claiming it started from scratch.
    """
    module = _script()
    paths = {name: tmp_path / f"{name}.jsonl" for name in ("eager", "mixed", "compiled")}
    for path, records in zip(paths.values(), _chain(module, **_EARNED), strict=True):
        _write_report(path, records)
    run_directory = tmp_path / "run"
    artifact = tmp_path / "bc-actor.pt"
    artifact.write_bytes(b"cloned actor")
    invocation: dict[str, object] = {}

    class Executed(Exception):
        pass

    def fake_execv(executable: str, command: list[str]) -> None:
        invocation.update(executable=executable, command=command)
        raise Executed

    monkeypatch.setattr(module.os, "execv", fake_execv)
    argv = [
        "launch_calibrated_training.py",
        "--eager-report",
        str(paths["eager"]),
        "--mixed-report",
        str(paths["mixed"]),
        "--compiled-report",
        str(paths["compiled"]),
        "--run-dir",
        str(run_directory),
        "--iterations",
        "500",
        "--init-actor-from",
        str(artifact),
        "--critic-warmup-iterations",
        "15",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(Executed):
        module.main()

    decision_path = run_directory / "calibration-decision.json"
    first = json.loads(decision_path.read_text())
    assert first["initial_actor"] == str(artifact)
    assert first["critic_warmup_iterations"] == 15
    assert first["resume_checkpoint"] is None
    command = first["training_command"]
    assert command[command.index("--init-actor-from") + 1] == str(artifact)
    assert command[command.index("--critic-warmup-iterations") + 1] == "15"

    # The run crashed inside the warmup window and left a checkpoint behind.
    latest_checkpoint = run_directory / "latest.pt"
    latest_checkpoint.write_bytes(b"atomic checkpoint")

    with pytest.raises(Executed):
        module.main()

    second = json.loads(decision_path.read_text())
    assert second["resume_checkpoint"] == str(latest_checkpoint)
    assert second["initial_actor"] == str(artifact)
    assert second["critic_warmup_iterations"] == 15
    resumed = second["training_command"]
    assert invocation["command"] == resumed
    assert "--init-actor-from" not in resumed
    assert "--critic-warmup-iterations" not in resumed
    assert resumed[-2:] == ["--resume", str(latest_checkpoint)]

    training = _script("train_ppo.py")
    monkeypatch.setattr(sys, "argv", ["train_ppo.py", *resumed[2:]])
    arguments = training.parse_args()
    training._validate_args(arguments)
    assert arguments.resume == latest_checkpoint
    assert arguments.init_actor_from is None
    assert arguments.critic_warmup_iterations is None


def test_a_warmup_that_outlasts_the_run_is_rejected(tmp_path: Path) -> None:
    """The actor would never take a step, and the stalled-actor guard is
    suppressed for exactly those iterations, so the run would finish silently
    identical to the clone it started from."""
    with pytest.raises(ValueError, match="leave iterations for the actor"):
        build_training_command(
            tmp_path / "run",
            iterations=15,
            max_hours=0.0,
            seed=7,
            rollout_forward_mode="eager",
            update_compile_mode="eager",
            initial_actors=(tmp_path / "bc-actor.pt",),
            critic_warmup_iterations=15,
        )


def test_the_chain_fixture_declares_every_key_a_real_benchmark_emits(
    monkeypatch, tmp_path: Path
) -> None:
    """`_records` writes a configuration record by hand, and that is how a real
    breakage hid in this file.

    `_comparable_configuration` removes the knobs and requires every remaining
    key to be identical across the three nodes, so a key the benchmark emits and
    this fixture omits is enforced by the launcher and exercised by nothing here.
    When the collection mode and precision were added to the report, every real
    chain began failing with "eager and compiled benchmark configurations differ
    at ['rollout_bfloat16', 'rollout_forward_mode']" while every chain test above
    stayed green, because the fixture had never heard of either key.

    So the fixture is checked against the script rather than against itself. Only
    the key set: the values differ by design (this fixture runs cuda medians the
    benchmark below cannot produce on cpu), and it is the key set that decides
    what the launcher compares.
    """
    module = _script()
    benchmark = _script("benchmark_ppo_iteration.py")

    class Collected(Exception):
        """Ends the benchmark at its first collection, after the record is written."""

    def collect(*_arguments: object, **_keywords: object) -> None:
        raise Collected

    monkeypatch.setattr(benchmark, "collect_mixed_play_rust", collect)
    report = tmp_path / "benchmark.jsonl"
    # A one-game cpu wave with a small model: the configuration record is emitted
    # before any of the timing happens, and the knobs are inert off cuda anyway.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_ppo_iteration.py",
            "--device",
            "cpu",
            "--games",
            "1",
            "--league-games",
            "2",
            "--league-opponents",
            "2",
            "--repeats",
            "2",
            "--cnn-width",
            "8",
            "--cnn-blocks",
            "1",
            "--model-dim",
            "16",
            "--transformer-layers",
            "3",
            "--output",
            str(report),
        ],
    )
    try:
        with pytest.raises(Collected):
            benchmark.main()
    finally:
        benchmark._configure_report(None)

    emitted = json.loads(report.read_text(encoding="utf-8").splitlines()[0])
    fixture = _records(
        module, rollout_forward_mode="inductor", update_compile_mode="default", seconds=10.0
    )[0]

    assert emitted["event"] == "configuration"
    assert set(emitted) == set(fixture)
