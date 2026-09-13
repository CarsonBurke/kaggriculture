#!/usr/bin/env python3
"""Audit sampling-vs-update replay parity for a trained actor before launch.

The calibration benchmark also runs `update_replay_parity`, but it does so on a
freshly initialized actor, whose heads are near-uniform.  The divergence this
audits is a function of head sharpness: a random-init actor measures a
per-component KL around 4e-7 where a behavior-cloned one measures around 2e-3,
four orders of magnitude apart.  So calibration alone cannot tell you whether a
*pretrained* actor will clear the gate that training enforces, and a warm start
that fails does so at its first iteration, after the run has already been
queued, compiled and staged.

This runs the shipped gate on the shipped code path -- the same
`update_replay_parity`, the same production minibatch size, autocast and
compilation settings, over a production-sized mixed rollout -- across several
waves, and reports the two gated statistics against their bounds.  Several
waves because one is not evidence of a margin: the statistic is a mean over
hundreds of thousands of components and its spread across independent
environments is what decides whether a measured value sits safely inside the
bound or merely happened to.

What this cannot tell you: the league opponents here are frozen copies of the
actor under audit, which is what iteration zero of a warm start actually plays,
but training enforces the same bound at iterations 25, 50 and beyond, where the
league holds genuinely older snapshots.  The divergence is a function of head
sharpness on the states visited, and different opponents visit different
states, so a passing audit is evidence about the launch iteration rather than a
guarantee for the whole run.  That is what the in-training cadence is for.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch._dynamo

from kaggriculture.inference import load_actor_artifact
from kaggriculture.ppo import (
    MAX_FIRST_MINIBATCH_KL,
    MAX_UPDATE_REPLAY_KL,
    MAX_UPDATE_REPLAY_TAIL_FRACTION,
    UPDATE_COMPILE_MODES,
    UPDATE_REPLAY_TAIL_LOGPROB,
    PpoConfig,
    update_replay_parity,
)
from kaggriculture.production import (
    PRODUCTION_EPISODE_STEPS,
    PRODUCTION_LEAGUE_ACTIVE_OPPONENTS,
    PRODUCTION_LEAGUE_GAMES,
    PRODUCTION_LEAGUE_HISTORICAL_OPPONENTS,
    PRODUCTION_SELF_PLAY_GAMES,
    PRODUCTION_TEMPERATURE,
    production_ppo_config,
)
from kaggriculture.provenance import source_identity
from kaggriculture.registry import resolve_architecture
from kaggriculture.rollout import (
    ROLLOUT_FORWARD_MODES,
    allocate_rollout_storage,
    collect_mixed_play_rust,
)

COMPONENTS = ("unit", "kind", "quantity")
_PRODUCTION_LEAGUE_OPPONENTS = (
    PRODUCTION_LEAGUE_ACTIVE_OPPONENTS + PRODUCTION_LEAGUE_HISTORICAL_OPPONENTS
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--actor",
        type=Path,
        required=True,
        help="actor artifact to audit, e.g. a behavior-cloning warm start",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=20260812,
        help="first rollout seed; the rest are spaced by one wave's seed span",
    )
    parser.add_argument(
        "--measurements",
        type=int,
        default=3,
        help="independent rollouts to measure; the spread across them is the evidence",
    )
    parser.add_argument("--self-play-games", type=int, default=PRODUCTION_SELF_PLAY_GAMES)
    parser.add_argument("--league-games", type=int, default=PRODUCTION_LEAGUE_GAMES)
    parser.add_argument("--episode-steps", type=int, default=PRODUCTION_EPISODE_STEPS)
    parser.add_argument("--temperature", type=float, default=PRODUCTION_TEMPERATURE)
    parser.add_argument("--device", default="cuda")
    # The same knobs train_ppo.py takes, because this audit is only evidence
    # about a launch it matches exactly. Required rather than defaulted: with a
    # default, `--actor X` alone would audit some pairing nothing trains in and
    # exit zero, and nothing downstream reads this report to catch that -- the
    # same failure the cuda guard below refuses for the same reason.
    #
    # Both compile knobs are modes, and neither has a boolean here. A boolean
    # cannot name a three- or five-valued decision, and letting a mode and a
    # boolean be set independently would allow auditing an Inductor learner
    # against an eager frozen league ensemble: a mix train_ppo.py cannot produce,
    # because it derives the ensemble decision from the mode. An audit that can
    # express unlaunchable pairings is not evidence about a launch.
    parser.add_argument("--rollout-bfloat16", action=argparse.BooleanOptionalAction, required=True)
    # The knob with the largest measured effect on collection cost: `cudagraphs`
    # is slower than not compiling at all in fp32 (5.309 ms against 4.907 ms on
    # the production wave) and `inductor` is 1.8x faster at 2.720 ms.
    parser.add_argument(
        "--rollout-forward-mode",
        choices=ROLLOUT_FORWARD_MODES,
        required=True,
    )
    # No default for the same reason: an audit that can default the update knob
    # measures a pairing nothing trains in, and the modes differ in whether they
    # capture CUDA graphs and whether they benchmark kernel selection -- all of
    # which move the graphs whose divergence from the collector this measures.
    parser.add_argument(
        "--update-compile-mode",
        choices=UPDATE_COMPILE_MODES,
        required=True,
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="optional JSONL path recording every per-seed measurement",
    )
    return parser.parse_args()


def _seeds(args: argparse.Namespace) -> list[int]:
    """Return wave seeds far enough apart to draw disjoint environments.

    One wave consumes a contiguous block of environment seeds -- one per
    physical game -- so hand-picked seeds a few hundred apart share most of
    their environments and the spread across them measures almost nothing.
    Spacing by the full span is what makes each measurement independent.
    """
    if args.measurements < 1:
        raise ValueError("at least one measurement is required")
    span = args.self_play_games + args.league_games
    return [args.base_seed + index * span for index in range(args.measurements)]


def _measure(
    args: argparse.Namespace,
    actor: torch.nn.Module,
    model_config: dict[str, int | float],
    architecture_name: str,
    ppo_config: PpoConfig,
    device: torch.device,
    arena: dict[str, np.ndarray],
    seed: int,
) -> dict[str, float | int]:
    # The steady-state league composition -- a full opponent slate, every seat
    # decoding at the learner's own temperature -- which is what training plays
    # from the moment the snapshot archive fills. Iteration zero of a warm start
    # is thinner than this: `select_league_mix` has only the initial snapshot to
    # draw on and returns a single active opponent. Auditing the steady state is
    # the right choice because training enforces the same bound at every later
    # audit, and the weights are the actor's own either way.
    #
    # There are no per-lane temperature or determinism arrays here, and their
    # absence is the point: they existed to reproduce production's split between
    # stochastic active lanes and argmax historical ones, and production no
    # longer runs that wave. Keeping them would make this audit evidence about a
    # launch that does not exist.
    opponents: list[torch.nn.Module] = []
    assignments = None
    if args.league_games:
        state = actor.state_dict()
        for _ in range(_PRODUCTION_LEAGUE_OPPONENTS):
            opponent = resolve_architecture(architecture_name).build_actor(model_config)
            opponent.load_state_dict(state)
            opponent.requires_grad_(False)
            opponents.append(opponent.to(device).eval())
        generator = np.random.default_rng(seed)
        assignments = np.arange(args.league_games, dtype=np.int64) % len(opponents)
        generator.shuffle(assignments)

    rollout = collect_mixed_play_rust(
        actor,
        opponents,
        self_play_games=args.self_play_games,
        league_games=args.league_games,
        opponent_indices=assignments,
        seed_start=seed,
        episode_steps=args.episode_steps,
        temperature=args.temperature,
        opponent_temperature=args.temperature,
        sampling_seed=seed ^ 0x5EED,
        # The sampling forward is one of the two sides this audit measures, so
        # it takes the collection knobs and the update below takes the update
        # one. Whichever mode and precision measurement selects is the pairing
        # production runs, so it is the one whose divergence must be audited.
        # Matching the update path's bf16 is not the riskier choice: on this
        # actor it measures 8.4x LOWER drift than fp32 collection, because the
        # drift is dominated by systematic differences between the two paths
        # rather than by rounding noise. It still moves the sampled policy, so
        # it is measured rather than assumed.
        #
        # The frozen league ensemble follows the learner, exactly as train_ppo.py
        # derives it, so this audit cannot express a pairing a launch cannot.
        forward_mode=args.rollout_forward_mode,
        forward_autocast=args.rollout_bfloat16,
        storage=arena,
    )
    del opponents
    return update_replay_parity(
        actor,
        rollout,
        minibatch_size=ppo_config.minibatch_size,
        compile_mode=ppo_config.update_compile_mode,
        autocast_enabled=ppo_config.use_bfloat16 and device.type == "cuda",
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    # Compilation and autocast are part of what is being audited: roughly 138x
    # of the divergence is bf16 in the update forward, so a CPU run measures a
    # number two orders of magnitude low and would exit zero having audited a
    # configuration nothing will ever train in.
    if device.type != "cuda":
        raise SystemExit(f"replay parity must be audited on cuda, not {device.type}")
    actor, payload = load_actor_artifact(args.actor, device=device)
    actor.eval()
    architecture_name = resolve_architecture(payload).name
    # The launcher chooses compilation from a measured speedup rather than
    # unconditionally, so the audit has to be able to follow it either way.
    ppo_config = PpoConfig(**production_ppo_config(update_compile_mode=args.update_compile_mode))

    # One arena for the whole audit, exactly as training holds one for the
    # whole run. A defect in "every field is rewritten each wave" leaves the
    # previous wave's values behind, which is a large parity divergence and
    # among the very defects this exists to catch -- a fresh allocation per
    # measurement would make that class unobservable.
    arena = allocate_rollout_storage(
        architecture_name,
        args.self_play_games * 2 + args.league_games,
        args.episode_steps - 1,
        pin_memory=device.type == "cuda",
    )
    records: list[dict[str, object]] = []
    try:
        for seed in _seeds(args):
            # No compilation reset between waves: `valid` is set all-True by
            # the native storage and never cleared, so every wave partitions
            # the same trajectories * horizon count into the same minibatch
            # shapes. Resetting would only discard reusable Inductor graphs
            # and pay a cold compile per wave.
            torch.cuda.empty_cache()
            metrics = _measure(
                args,
                actor,
                payload["model_config"],
                architecture_name,
                ppo_config,
                device,
                arena,
                seed,
            )
            records.append({"seed": seed, **metrics})
            print(
                f"seed {seed}: kl {float(metrics['update_replay_max_kl']):.4e}"
                f"  tail_fraction {float(metrics['update_replay_max_tail_fraction']):.4e}"
                f"  max|exp(d)-1| {float(metrics['update_replay_max_ratio_error']):.4f}"
            )
    finally:
        # Whatever was measured before a failure is still the evidence for
        # what failed, so the report is written on every exit path. The
        # configuration record carries the source identity because a parity
        # measurement is only evidence about the tree that produced it, and
        # every other gate artifact in this tree is stamped the same way.
        if args.report is not None and records:
            configuration = {
                "event": "configuration",
                "actor": str(args.actor),
                "self_play_games": args.self_play_games,
                "league_games": args.league_games,
                "episode_steps": args.episode_steps,
                "rollout_forward_mode": args.rollout_forward_mode,
                "rollout_bfloat16": args.rollout_bfloat16,
                "update_compile_mode": ppo_config.update_compile_mode,
                "use_bfloat16": ppo_config.use_bfloat16,
                "minibatch_size": ppo_config.minibatch_size,
                "max_update_replay_kl": MAX_UPDATE_REPLAY_KL,
                "max_update_replay_tail_fraction": MAX_UPDATE_REPLAY_TAIL_FRACTION,
                "max_first_minibatch_kl": MAX_FIRST_MINIBATCH_KL,
                "update_replay_tail_logprob": UPDATE_REPLAY_TAIL_LOGPROB,
                "source_identity": source_identity(),
            }
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("w", encoding="utf-8") as handle:
                for record in (configuration, *records):
                    handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    for record in records:
        for component in COMPONENTS:
            if record[f"update_replay_{component}_active_count"] < 1:
                raise SystemExit(f"update replay parity saw no active {component} components")
    worst_kl = max(float(record["update_replay_max_kl"]) for record in records)
    worst_tail = max(float(record["update_replay_max_tail_fraction"]) for record in records)
    # The statistic `update_ppo` gates every iteration. It is audited here
    # because a bound is only as good as the actor it was calibrated on, and
    # this one was calibrated on a randomly initialized actor whose logits are
    # nearly uniform -- a regime in which the bf16 update forward diverges by
    # ~7e-8 and every warm-started actor since has diverged by ~2e-3. Auditing
    # the per-head means while leaving their component-weighted average
    # ungoverned is what let a 50x-wrong bound survive two recalibrations of
    # its own siblings.
    worst_joint = max(float(record["update_replay_joint_kl"]) for record in records)
    worst_component = max(float(record["update_replay_component_kl"]) for record in records)
    worst_minibatch = max(float(record["update_replay_minibatch_kl"]) for record in records)
    # Same sampler likelihoods, but the grad-tracking update graph rather than
    # the no-grad replay graph. The optimizer never rewrites the behavior side.
    worst_first = max(float(record["update_replay_first_minibatch_kl"]) for record in records)
    worst_first_mean = max(float(record["update_replay_mean_minibatch_kl"]) for record in records)
    print(
        f"\nworst over {len(records)} waves: kl {worst_kl:.4e} "
        f"(bound {MAX_UPDATE_REPLAY_KL}, margin {MAX_UPDATE_REPLAY_KL / max(worst_kl, 1e-30):.1f}x)"
        f"  tail_fraction {worst_tail:.4e} (bound {MAX_UPDATE_REPLAY_TAIL_FRACTION}, "
        f"margin {MAX_UPDATE_REPLAY_TAIL_FRACTION / max(worst_tail, 1e-30):.1f}x)"
        f"\njoint kl diagnostic {worst_joint:.4e}; component kl {worst_component:.4e}; "
        f"worst single minibatch {worst_minibatch:.4e}, minibatch-to-batch ratio "
        f"{worst_minibatch / max(worst_component, 1e-30):.2f}x"
        f"\nsampler-vs-update worst minibatch {worst_first:.4e} "
        f"(bound {MAX_FIRST_MINIBATCH_KL}, margin "
        f"{MAX_FIRST_MINIBATCH_KL / max(worst_first, 1e-30):.1f}x)"
        f", mean over minibatches {worst_first_mean:.4e}, "
        f"tail ratio {worst_first / max(worst_first_mean, 1e-30):.1f}x"
    )

    if worst_kl > MAX_UPDATE_REPLAY_KL:
        raise SystemExit(
            f"sampling-vs-update policy divergence exceeded {MAX_UPDATE_REPLAY_KL}: {worst_kl}"
        )
    if worst_tail > MAX_UPDATE_REPLAY_TAIL_FRACTION:
        raise SystemExit(
            "sampling-vs-update materially divergent component share exceeded "
            f"{MAX_UPDATE_REPLAY_TAIL_FRACTION}: {worst_tail}"
        )
    if worst_first > MAX_FIRST_MINIBATCH_KL:
        raise SystemExit(
            "single-minibatch sampler-vs-update divergence exceeded "
            f"{MAX_FIRST_MINIBATCH_KL}: {worst_first}"
        )


if __name__ == "__main__":
    main()
