from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kaggriculture.provenance import source_identity


def _report_records(report: Path) -> list[dict]:
    return [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]


def _audit_script():
    path = Path(__file__).parents[1] / "scripts" / "audit_replay_parity.py"
    spec = importlib.util.spec_from_file_location("kaggriculture_audit_replay_parity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _passing_metrics() -> dict[str, float | int]:
    # The magnitudes are the behavior-cloned conv actor's measured ones rather
    # than round placeholders. A stub twenty times below what the tree documents
    # as healthy passes every bound trivially, which is how a gate calibrated
    # against a randomly initialized actor survived in a suite that was green.
    return {
        "update_replay_max_kl": 1.9e-3,
        "update_replay_max_tail_fraction": 3.2e-5,
        "update_replay_max_ratio_error": 61.5,
        # A component-weighted mean of the heads cannot exceed the largest of
        # them, and the worst minibatch cannot fall below that mean.
        "update_replay_joint_kl": 1.7e-3,
        "update_replay_minibatch_kl": 5.0e-3,
        # A different comparison entirely: replay against update forward.
        "update_replay_first_minibatch_kl": 5.0e-3,
        "update_replay_mean_minibatch_kl": 2.0e-3,
        **{
            f"update_replay_{component}_active_count": 1000
            for component in ("unit", "kind", "quantity")
        },
    }


def test_measurement_seeds_draw_disjoint_environment_blocks() -> None:
    module = _audit_script()
    args = SimpleNamespace(
        base_seed=1000,
        measurements=4,
        self_play_games=112,
        league_games=96,
    )
    seeds = module._seeds(args)

    # One wave consumes one environment seed per physical game, contiguously
    # from its start seed, so consecutive measurements must be at least a whole
    # wave apart or they replay each other's environments.
    span = args.self_play_games + args.league_games
    assert seeds == [1000, 1208, 1416, 1624]
    blocks = [set(range(seed, seed + span)) for seed in seeds]
    for index, block in enumerate(blocks):
        for other in blocks[index + 1 :]:
            assert not block & other

    with pytest.raises(ValueError, match="at least one measurement"):
        module._seeds(SimpleNamespace(**{**vars(args), "measurements": 0}))


def _run(
    monkeypatch,
    module,
    tmp_path: Path,
    measure,
    *extra: str,
    pairing: tuple[str, ...] = (
        "--update-compile-mode",
        "eager",
        "--no-rollout-bfloat16",
        "--rollout-forward-mode",
        "eager",
    ),
) -> Path:
    # The audit refuses to assume either knob, so every invocation states the
    # pairing. Tests about something else state the all-eager one explicitly
    # rather than relying on a default, which is the whole point: there isn't
    # one. `pairing` is separate from `extra` so a test can pass a partial or
    # empty pairing to exercise the refusal.
    report = tmp_path / "nested" / "parity.jsonl"
    monkeypatch.setattr(
        module,
        "load_actor_artifact",
        lambda path, device=None: (SimpleNamespace(eval=lambda: None), {"model_config": {}}),
    )
    monkeypatch.setattr(module, "resolve_architecture", lambda payload: SimpleNamespace(name="x"))
    monkeypatch.setattr(module, "allocate_rollout_storage", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "_measure", measure)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_replay_parity.py",
            "--actor",
            str(tmp_path / "actor.pt"),
            "--device",
            "cuda",
            "--measurements",
            "3",
            "--report",
            str(report),
            *pairing,
            *extra,
        ],
    )
    return report


def test_the_audit_records_the_phase_configuration_it_actually_measured(
    monkeypatch,
    tmp_path,
) -> None:
    # The launcher chooses each knob from a measured speedup rather than
    # unconditionally, and the collection forward's mode and precision move the
    # gated divergence by more than an order of magnitude in either direction,
    # so a report that did not say which paths it measured would be evidence
    # about a configuration nothing necessarily trains in. The last case is the
    # one measurement selects on the conv model: a collector matching the
    # update path's Inductor bf16, which is both the fastest and the lowest
    # drift precisely because it matches.
    module = _audit_script()

    for index, (flags, update, mode, bf16) in enumerate(
        (
            (
                (
                    "--update-compile-mode",
                    "eager",
                    "--no-rollout-bfloat16",
                    "--rollout-forward-mode",
                    "eager",
                ),
                "eager",
                "eager",
                False,
            ),
            (
                (
                    "--update-compile-mode",
                    "default",
                    "--no-rollout-bfloat16",
                    "--rollout-forward-mode",
                    "cudagraphs",
                ),
                "default",
                "cudagraphs",
                False,
            ),
            (
                (
                    "--update-compile-mode",
                    "max-autotune",
                    "--rollout-bfloat16",
                    "--rollout-forward-mode",
                    "inductor",
                ),
                "max-autotune",
                "inductor",
                True,
            ),
        )
    ):
        report = _run(
            monkeypatch,
            module,
            tmp_path / str(index),
            lambda *a, **k: _passing_metrics(),
            pairing=flags,
        )
        module.main()
        configuration, *_records = _report_records(report)

        assert configuration["update_compile_mode"] == update
        assert configuration["rollout_forward_mode"] == mode
        assert configuration["rollout_bfloat16"] is bf16
        assert configuration["use_bfloat16"] is True


@pytest.mark.parametrize(
    "flags",
    [
        (),
        ("--update-compile-mode", "default"),
        # The collection knobs are the same kind of assumption: they move the
        # gated statistic further than the update knob does.
        ("--update-compile-mode", "eager", "--no-rollout-bfloat16"),
        ("--update-compile-mode", "eager", "--rollout-forward-mode", "eager"),
        # A mode outside its knob's domain is the same defect stated positively:
        # the two domains are disjoint, so each knob wearing the other's mode
        # names a pairing no launch can produce. argparse `choices` is what
        # refuses it, at the only boundary that knows both domains.
        (
            "--update-compile-mode",
            "cudagraphs",
            "--no-rollout-bfloat16",
            "--rollout-forward-mode",
            "eager",
        ),
        (
            "--update-compile-mode",
            "eager",
            "--no-rollout-bfloat16",
            "--rollout-forward-mode",
            "max-autotune",
        ),
    ],
)
def test_the_audit_refuses_to_assume_a_pairing_it_was_not_told(monkeypatch, tmp_path, flags):
    # An unstated knob is the failure this audit exists to prevent, not a
    # convenience. A defaulted flag would make the zero-flag invocation -- the
    # one an operator reaches for -- measure some pairing and pass, having
    # audited a configuration nothing trains in. Nothing downstream reads this
    # report to catch that, so the CLI has to. Every knob must be stated;
    # stating only some is still an assumption about the rest.
    module = _audit_script()
    _run(monkeypatch, module, tmp_path, lambda *a, **k: _passing_metrics(), pairing=flags)

    with pytest.raises(SystemExit) as failure:
        module.main()

    assert failure.value.code != 0


@pytest.mark.parametrize("mode", ["eager", "cudagraphs", "inductor"])
def test_the_audit_cannot_express_a_pairing_a_launch_cannot_produce(
    monkeypatch,
    tmp_path,
    mode,
) -> None:
    # This audit once took a `--compile-rollout` boolean alongside the mode,
    # which let it certify an Inductor learner against an eager frozen league
    # ensemble -- a mix `train_ppo.py` cannot produce. The collector now has a
    # single execution knob and the ensemble follows it, so no second parameter
    # can disagree with the mode. Pinned here because the defect was invisible
    # to a suite in which two files asserted opposite things about the boolean.
    module = _audit_script()
    seen: dict[str, object] = {}

    def collect(*args, **kwargs):
        seen.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(module, "collect_mixed_play_rust", collect)
    _run(
        monkeypatch,
        module,
        tmp_path,
        module._measure,
        # No league seats: this test is about the derivation the collector call
        # receives, and building frozen opponents needs a real actor.
        "--league-games",
        "0",
        pairing=(
            "--update-compile-mode",
            "default",
            "--rollout-bfloat16",
            "--rollout-forward-mode",
            mode,
        ),
    )
    # The parsed namespace is where a second knob for either phase would show
    # up. Asserting on `parse_args` itself asserts nothing -- a function has no
    # such attribute either way -- which is how a stale boolean could survive a
    # test that looked like it forbade one.
    parsed = module.parse_args()
    assert not hasattr(parsed, "compile_rollout")
    assert not hasattr(parsed, "compile_update")
    with pytest.raises(SystemExit):
        module.main()

    assert seen["forward_mode"] == mode
    assert "compile_models" not in seen


def test_the_audit_refuses_a_device_that_would_measure_a_different_number(
    monkeypatch,
    tmp_path,
) -> None:
    # Roughly 138x of the divergence is bf16 in the update forward, so a CPU
    # run measures a number two orders of magnitude low and would exit zero
    # having audited a configuration nothing will ever train in.
    module = _audit_script()
    report = _run(monkeypatch, module, tmp_path, lambda *a, **k: _passing_metrics())
    argv = [value if value != "cuda" else "cpu" for value in sys.argv]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit, match="must be audited on cuda, not cpu"):
        module.main()

    assert not report.exists()


def test_a_divergent_measurement_fails_the_audit_and_still_records_its_evidence(
    monkeypatch,
    tmp_path,
) -> None:
    module = _audit_script()
    divergent = {
        **_passing_metrics(),
        "update_replay_max_kl": module.MAX_UPDATE_REPLAY_KL * 2.0,
    }
    measurements = [_passing_metrics(), divergent, _passing_metrics()]
    report = _run(monkeypatch, module, tmp_path, lambda *args, **kwargs: measurements.pop(0))

    with pytest.raises(SystemExit, match="policy divergence exceeded"):
        module.main()

    configuration, *records = _report_records(report)
    # A parity measurement is only evidence about the tree that produced it.
    assert configuration["source_identity"] == source_identity()
    assert configuration["max_update_replay_kl"] == module.MAX_UPDATE_REPLAY_KL
    assert len(records) == 3
    assert max(record["update_replay_max_kl"] for record in records) == pytest.approx(
        module.MAX_UPDATE_REPLAY_KL * 2.0
    )


def test_a_single_divergent_minibatch_fails_the_audit(monkeypatch, tmp_path) -> None:
    """The per-iteration gate's own statistic has to be audited on the clone.

    Every sampling-versus-replay mean can sit comfortably inside its bound while
    the replay-versus-update residual does not, and it is the latter that
    `update_ppo` checks every iteration. Auditing only the batch means is how a
    bound calibrated against a randomly initialized actor survived and stopped a
    run at its first actor update.
    """
    module = _audit_script()
    divergent = {
        **_passing_metrics(),
        "update_replay_first_minibatch_kl": module.MAX_FIRST_MINIBATCH_KL * 2.0,
    }
    measurements = [_passing_metrics(), divergent, _passing_metrics()]
    report = _run(monkeypatch, module, tmp_path, lambda *args, **kwargs: measurements.pop(0))

    with pytest.raises(SystemExit, match="single-minibatch"):
        module.main()

    configuration, *records = _report_records(report)
    assert configuration["max_first_minibatch_kl"] == module.MAX_FIRST_MINIBATCH_KL
    assert max(record["update_replay_first_minibatch_kl"] for record in records) == pytest.approx(
        module.MAX_FIRST_MINIBATCH_KL * 2.0
    )


def test_a_measurement_that_raises_still_leaves_the_completed_ones_on_disk(
    monkeypatch,
    tmp_path,
) -> None:
    # The audit exists to produce evidence; a crash on the third wave must not
    # discard what the first two measured, since those are what a diagnosis
    # would start from.
    module = _audit_script()
    remaining = [_passing_metrics(), _passing_metrics()]

    def measure(*args, **kwargs):
        if not remaining:
            raise RuntimeError("collector exploded")
        return remaining.pop(0)

    report = _run(monkeypatch, module, tmp_path, measure)

    with pytest.raises(RuntimeError, match="collector exploded"):
        module.main()

    _configuration, *records = _report_records(report)
    assert [record["seed"] for record in records] == module._seeds(
        SimpleNamespace(base_seed=20260812, measurements=2, self_play_games=112, league_games=96)
    )


def test_a_head_with_no_active_components_fails_rather_than_passing_vacuously(
    monkeypatch,
    tmp_path,
) -> None:
    module = _audit_script()
    vacuous = {**_passing_metrics(), "update_replay_kind_active_count": 0}
    _run(monkeypatch, module, tmp_path, lambda *args, **kwargs: dict(vacuous))

    with pytest.raises(SystemExit, match="no active kind components"):
        module.main()
