#!/usr/bin/env python3
"""Measure how causally a behavior clone's decisions depend on opponent features.

The clone of `public-v27` in `runs/bc4-v27-conv` earns 92 money against the
built-in `starter` agent but ~28,900 against a copy of itself and ~80,000
against the real teacher, while the teacher earns 128,318-178,674 against
`starter`.  The farming skill is therefore present but gated on something only
a v27-like opponent supplies.  This probe tests that hypothesis causally and
exactly, on states recorded by `scripts/extract_bc_dataset.py` rather than on
live rollouts, so it is CPU-only, deterministic, and free of any confound from
the clone's own trajectory drifting off-distribution.

Three conditions per state, with the legality masks held at their recorded
values in all three (the ablation changes what the network sees, never what the
engine permits):

  real          the observation exactly as recorded;
  zeroed        the opponent's contribution to the actor input set to zero;
  transplanted  the opponent's contribution replaced by that of a state drawn
                from the next dataset, cyclically, at the same episode step, so
                a poor-opponent state is shown a rich opponent and vice versa.
                Given one dataset the donor is that same corpus, which permutes
                the opponent within one opponent style instead of crossing
                styles; `donor_is_same_corpus` in the report records which.

`zeroed` is off the state manifold by construction -- an all-zero farm has no
on-board tiles at all -- so it bounds the sensitivity rather than describing a
reachable situation.  `transplanted` is the load-bearing condition: every input
it produces is one the encoder really emits, and `transplant_pairing` in the
report shows the paired states demonstrate the same teacher action, so any
`real` -> `transplanted` change in the policy is spurious rather than a
legitimate adaptation.

The unit and market-kind heads are probed.  The quantity head is conditioned on
an already-selected market kind, so the BUY_SEED signal is entirely in the kind
head and probing quantities would only re-measure it behind a teacher-forced
gate.
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kaggriculture.actions import (
    MarketKind,
    UnitAction,
    all_unit_action_masks,
    market_kind_mask,
)
from kaggriculture.encoding import (
    BOARD_CHANNELS,
    FARM_CHANNELS,
    GLOBAL_FEATURES,
    encode_observation,
)
from kaggriculture.inference import checkpoint_orientation, load_actor_artifact
from kaggriculture.orientation import Orientation
from kaggriculture.provenance import file_sha256
from kaggriculture.registry import CONV_ENTITY, architecture_of

PROBE_FORMAT_VERSION = 1

# --- Where the opponent lives in the conv-entity actor input -----------------
#
# `encode_observation` builds the board as
# `concatenate((_encode_farm(own_farm), _encode_farm(opponent_farm)), axis=0)`,
# with each farm contributing exactly `FARM_CHANNELS` planes and the pair
# summing to `BOARD_CHANNELS`.  The opponent is therefore the upper half of the
# channel axis, exactly.
OPPONENT_BOARD_CHANNELS = slice(FARM_CHANNELS, BOARD_CHANNELS)

# The global vector opens with six clock features (day, hour, step, steps
# remaining, sin(hour), cos(hour)), then `for farm in (own_farm,
# opponent_farm)` appends four features each: money, unlocked quadrants, hands,
# hires today.  Everything after that pair is the acting player's own private
# state, shared market state, or shared town state.  These two counts are the
# only literals in this file that name a layout; `verify_opponent_slices`
# re-derives both slices differentially from the encoder itself and fails the
# run if either is wrong or incomplete.
CLOCK_FEATURES = 6
PER_FARM_FEATURES = 4
OWN_GLOBAL_SLICE = slice(CLOCK_FEATURES, CLOCK_FEATURES + PER_FARM_FEATURES)
OPPONENT_GLOBAL_SLICE = slice(
    CLOCK_FEATURES + PER_FARM_FEATURES, CLOCK_FEATURES + 2 * PER_FARM_FEATURES
)
# `units`/`unit_positions` are built only from `own_farm`'s farmer and hands and
# the acting player's own `private.inventories`, so they carry no opponent
# contribution and are never ablated.  The market inventory/price and town
# features are shared world state that both players move; they describe the
# world rather than the opponent and are likewise left untouched, which makes
# every number below a lower bound on total opponent dependence.

# Action families resolved from the name tables rather than from integer ids.
# IntEnum iteration skips the readability aliases at the bottom of `UnitAction`,
# so each family is exactly its distinct members.  `productive` is the
# PLANT/WATER/HARVEST core; `husbandry` and `land` cover the remaining
# value-creating unit actions so no probability mass the ablations move can hide
# in an unreported family.
UNIT_FAMILIES: dict[str, tuple[int, ...]] = {
    "plant": tuple(action for action in UnitAction if action.name.startswith("PLANT_")),
    "water": (int(UnitAction.WATER),),
    "harvest": (int(UnitAction.HARVEST),),
    "husbandry": (
        int(UnitAction.FEED),
        int(UnitAction.CARE),
        int(UnitAction.FERTILIZE),
        int(UnitAction.COLLECT_FERTILIZER),
        *(action for action in UnitAction if action.name.startswith("PLACE_")),
    ),
    "land": (
        int(UnitAction.DIG),
        int(UnitAction.BUILD_COOP),
        int(UnitAction.BUILD_PASTURE),
    ),
    "pass": (int(UnitAction.PASS),),
}
UNIT_FAMILIES["productive"] = (
    *UNIT_FAMILIES["plant"],
    *UNIT_FAMILIES["water"],
    *UNIT_FAMILIES["harvest"],
)
MARKET_FAMILIES: dict[str, tuple[int, ...]] = {
    "buy_seed": tuple(kind for kind in MarketKind if kind.name.startswith("BUY_SEED_")),
    "sell": tuple(kind for kind in MarketKind if kind.name.startswith("SELL_")),
    "stop": (int(MarketKind.STOP),),
}

CONDITIONS = ("real", "zeroed", "transplanted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True, help="BC actor artifact (.pt)")
    parser.add_argument(
        "--datasets",
        type=Path,
        nargs="+",
        required=True,
        help="extract_bc_dataset output dirs; the transplant donor is the next one, cyclically",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON report to write")
    parser.add_argument(
        "--episodes-per-dataset",
        type=int,
        default=40,
        help="episode-seat archives sampled per dataset, spread evenly over the directory",
    )
    parser.add_argument(
        "--states-per-episode",
        type=int,
        default=25,
        help="states per archive, on a fixed step grid shared by every dataset so transplants "
        "pair at an identical episode step",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="torch intra-op threads; this machine is shared, so the default is one",
    )
    return parser.parse_args()


# --- Dataset sampling -------------------------------------------------------


@dataclass(frozen=True)
class StateSample:
    """Encoded actor inputs plus recorded masks/targets for sampled states."""

    board: np.ndarray  # [N, BOARD_CHANNELS, B, B] float16
    global_features: np.ndarray  # [N, GLOBAL_FEATURES] float16
    units: np.ndarray  # [N, MAX_UNITS, UNIT_FEATURES] float16
    unit_positions: np.ndarray  # [N, MAX_UNITS, 2] int64
    unit_masks: np.ndarray  # [N, MAX_UNITS, N_UNIT_ACTIONS] bool
    unit_active: np.ndarray  # [N, MAX_UNITS] bool
    unit_actions: np.ndarray  # [N, MAX_UNITS] int64
    market_kind_masks: np.ndarray  # [N, MAX_MARKET_ORDERS, N_MARKET_KINDS] bool
    market_active: np.ndarray  # [N, MAX_MARKET_ORDERS] bool
    market_kinds: np.ndarray  # [N, MAX_MARKET_ORDERS] int64
    steps: np.ndarray  # [N] int64, episode step of each state
    episodes: tuple[str, ...]
    rows_per_archive: int


def _archive_paths(dataset: Path, count: int) -> tuple[list[Path], int, dict[str, Any] | None]:
    """Archives to sample, the corpus size, and the manifest when one was written.

    An extraction that was killed before writing `manifest.json` still leaves
    individually complete per-episode-seat archives, so the manifest is
    provenance here and not a prerequisite.
    """
    manifest: dict[str, Any] | None = None
    manifest_path = dataset / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths = [dataset / entry["file"] for entry in manifest["episodes"]]
    else:
        paths = sorted(dataset.glob("episode-*.npz"))
    if not paths:
        raise ValueError(f"{dataset}: no episode-seat archives found")
    if count < 1:
        raise ValueError("--episodes-per-dataset must be positive")
    # Spread the sample over the whole directory rather than taking a prefix:
    # consecutive seeds are consecutive games and the first N would correlate.
    picks = np.unique(np.linspace(0, len(paths) - 1, min(count, len(paths))).round().astype(int))
    return [paths[index] for index in picks], len(paths), manifest


def _step_grid(rows: int, count: int) -> np.ndarray:
    if not 0 < count <= rows:
        raise ValueError(f"--states-per-episode must be in 1..{rows}")
    return np.unique(np.linspace(0, rows - 1, count).round().astype(np.int64))


def load_states(paths: list[Path], *, states_per_episode: int) -> StateSample:
    """Encode a fixed step grid from a spread-out sample of one dataset's archives.

    Sampling the same grid in every dataset is what makes the transplant honest:
    a donor is always available at the recipient's exact episode step, so the
    opponent farm being pasted in is at the same stage of the game and the
    recipient's own clock features stay consistent with it.
    """
    parts: dict[str, list[np.ndarray]] = {}
    steps: list[np.ndarray] = []
    used: list[str] = []
    grid: np.ndarray | None = None
    rows_per_archive = 0
    for path in paths:
        with np.load(path) as archive:
            recorded = {
                name: archive[name]
                for name in (
                    "unit_masks",
                    "unit_active",
                    "unit_actions",
                    "market_kind_masks",
                    "market_active",
                    "market_kinds",
                )
            }
            raw = json.loads(zlib.decompress(archive["raw_json_zlib"].tobytes()))
        rows = len(raw["observations"])
        if rows != recorded["unit_actions"].shape[0]:
            raise ValueError(
                f"{path}: {rows} raw states but {recorded['unit_actions'].shape[0]} targets"
            )
        if grid is None:
            grid, rows_per_archive = _step_grid(rows, states_per_episode), rows
        elif rows != rows_per_archive:
            raise ValueError(f"{path}: {rows} states; the first archive had {rows_per_archive}")
        encoded = [encode_observation(raw["observations"][index]["observation"]) for index in grid]
        observed = np.asarray(
            [int(raw["observations"][index]["observation"]["step"]) for index in grid],
            dtype=np.int64,
        )
        if not np.array_equal(observed, grid):
            raise ValueError(f"{path}: row index and recorded step disagree")
        block = {
            "board": np.stack([row.board for row in encoded]),
            "global_features": np.stack([row.global_features for row in encoded]),
            "units": np.stack([row.units for row in encoded]),
            "unit_positions": np.stack([row.unit_positions for row in encoded]),
            **{name: value[grid] for name, value in recorded.items()},
        }
        active = np.stack([row.unit_active for row in encoded])
        if not np.array_equal(active, block["unit_active"]):
            raise ValueError(f"{path}: projection and encoding disagree on active units")
        for name, value in block.items():
            parts.setdefault(name, []).append(value)
        steps.append(observed)
        used.append(path.name)
    joined = {name: np.concatenate(value) for name, value in parts.items()}
    return StateSample(
        board=joined["board"],
        global_features=joined["global_features"],
        units=joined["units"],
        unit_positions=joined["unit_positions"].astype(np.int64),
        unit_masks=joined["unit_masks"],
        unit_active=joined["unit_active"],
        unit_actions=joined["unit_actions"].astype(np.int64),
        market_kind_masks=joined["market_kind_masks"],
        market_active=joined["market_active"],
        market_kinds=joined["market_kinds"].astype(np.int64),
        steps=np.concatenate(steps),
        episodes=tuple(used),
        rows_per_archive=rows_per_archive,
    )


# --- Slice derivation check --------------------------------------------------


def verify_opponent_slices(dataset: Path, rows: int) -> dict[str, Any]:
    """Differentially confirm what the opponent farm reaches, and what it does not.

    Re-encodes real observations with only `farms[opponent]` swapped for another
    state's opponent farm, and separately with only `farms[player]` swapped, and
    asserts each perturbation moves nothing outside its derived slice.  This
    turns the two layout counts above into a checked claim.

    It also re-derives the legality masks after the opponent swap.  Both mask
    builders read `farms[player]`, the acting player's own `private`, and the
    shared market and town only, so the recorded masks stay exactly valid under
    every ablation; that is what makes holding them fixed correct rather than a
    stale-support bias in the reported masses and KLs.
    """
    path = _archive_paths(dataset, 1)[0][0]
    with np.load(path) as archive:
        raw = json.loads(zlib.decompress(archive["raw_json_zlib"].tobytes()))
    grid = _step_grid(rows, min(12, rows))
    entries = [raw["observations"][index]["observation"] for index in grid]
    moved: dict[str, set[int]] = {"opponent_board": set(), "opponent_global": set()}
    own_moved: dict[str, set[int]] = {"own_board": set(), "own_global": set()}
    for index, observation in enumerate(entries):
        base = encode_observation(observation)
        base_unit_masks = all_unit_action_masks(observation)
        base_kind_mask = market_kind_mask(observation)
        player = int(observation["player"])
        # Two donors per state, one nearby and one from far away in the episode,
        # so a feature that happens to agree with one donor still gets moved.
        for offset in (1, len(entries) // 2):
            donor = entries[(index + offset) % len(entries)]
            for role, sink in (("opponent", moved), ("own", own_moved)):
                seat = player if role == "own" else 1 - player
                farms = list(observation["farms"])
                farms[seat] = donor["farms"][seat]
                perturbed = {**observation, "farms": farms}
                probe = encode_observation(perturbed)
                board_delta = np.flatnonzero(np.any(probe.board != base.board, axis=(1, 2)))
                global_delta = np.flatnonzero(probe.global_features != base.global_features)
                sink[f"{role}_board"].update(board_delta.tolist())
                sink[f"{role}_global"].update(global_delta.tolist())
                if role != "opponent":
                    continue
                # Unit tokens come from the acting player's own farmer, hands,
                # and private inventories, so an opponent swap must not reach
                # them; that is what makes leaving them unablated correct.
                if not (
                    np.array_equal(probe.units, base.units)
                    and np.array_equal(probe.unit_positions, base.unit_positions)
                ):
                    raise AssertionError("opponent farm swap moved the acting player's unit tokens")
                if not np.array_equal(all_unit_action_masks(perturbed), base_unit_masks):
                    raise AssertionError("opponent farm swap changed the unit legality mask")
                if not np.array_equal(market_kind_mask(perturbed), base_kind_mask):
                    raise AssertionError("opponent farm swap changed the market legality mask")
    own_channels = set(range(FARM_CHANNELS))
    opponent_channels = set(range(FARM_CHANNELS, BOARD_CHANNELS))
    checks = (
        (moved["opponent_board"], opponent_channels, "opponent board channels"),
        (
            moved["opponent_global"],
            set(range(*OPPONENT_GLOBAL_SLICE.indices(GLOBAL_FEATURES))),
            "opponent global features",
        ),
        (own_moved["own_board"], own_channels, "own board channels"),
        (
            own_moved["own_global"],
            set(range(*OWN_GLOBAL_SLICE.indices(GLOBAL_FEATURES))),
            "own global features",
        ),
    )
    for observed, allowed, label in checks:
        if not observed <= allowed:
            raise AssertionError(f"{label}: perturbation moved {sorted(observed - allowed)}")
    if not moved["opponent_board"] or not moved["opponent_global"]:
        raise AssertionError("opponent perturbation moved nothing; the probe would be vacuous")
    return {
        "board_channels": [FARM_CHANNELS, BOARD_CHANNELS],
        "global_features": [
            OPPONENT_GLOBAL_SLICE.start,
            OPPONENT_GLOBAL_SLICE.stop,
        ],
        "observed_opponent_board_channels": sorted(moved["opponent_board"]),
        "observed_opponent_global_features": sorted(moved["opponent_global"]),
        "observed_own_board_channels": sorted(own_moved["own_board"]),
        "observed_own_global_features": sorted(own_moved["own_global"]),
        "probe_archive": path.name,
    }


# --- Ablations --------------------------------------------------------------


def donor_order(
    recipient: StateSample, donor: StateSample, generator: np.random.Generator
) -> np.ndarray:
    """For each recipient state, a donor state recorded at the same episode step.

    Same-step pairing keeps the pasted opponent farm at the recipient's stage of
    the game, so the measured effect is the opponent's wealth and development
    rather than a farm from a different phase of the episode.  When the donor is
    the recipient's own corpus (a single-dataset run) a state is never paired
    with itself, which would make the transplant a no-op for that row.
    """
    same_corpus = donor is recipient
    by_step: dict[int, np.ndarray] = {
        int(step): np.flatnonzero(donor.steps == step) for step in np.unique(donor.steps)
    }
    chosen = np.empty(recipient.steps.shape[0], dtype=np.int64)
    for index, step in enumerate(recipient.steps):
        candidates = by_step.get(int(step), np.empty(0, dtype=np.int64))
        if same_corpus:
            candidates = candidates[candidates != index]
        if candidates.size == 0:
            raise ValueError(f"donor dataset has no usable state at step {int(step)}")
        chosen[index] = candidates[generator.integers(candidates.size)]
    return chosen


def pairing_report(
    recipient: StateSample, donor: StateSample, order: np.ndarray
) -> dict[str, float]:
    """How far apart the paired states are, and whether they share a target.

    These decide how the transplant may be read.  If the recipient and its
    step-matched donor already demonstrate the same action, then the correct
    output is invariant to the opponent slice and any transplant-induced change
    in the policy is spurious by construction rather than a legitimate
    adaptation -- and, in the other direction, replacing the opponent slice is a
    label-preserving augmentation.  The own-slice distances say how much of the
    cross-dataset comparison is carried by the opponent at all.
    """
    paired = donor.board[order].astype(np.float32)
    paired_features = donor.global_features[order].astype(np.float32)
    own = recipient.board.astype(np.float32)
    own_features = recipient.global_features.astype(np.float32)
    both_units = recipient.unit_active & donor.unit_active[order]
    both_market = recipient.market_active & donor.market_active[order]
    return {
        "own_board_mean_abs_delta": float(
            np.abs(own[:, :FARM_CHANNELS] - paired[:, :FARM_CHANNELS]).mean()
        ),
        "own_global_mean_abs_delta": float(
            np.abs(own_features[:, OWN_GLOBAL_SLICE] - paired_features[:, OWN_GLOBAL_SLICE]).mean()
        ),
        "opponent_board_mean_abs_delta": float(
            np.abs(own[:, OPPONENT_BOARD_CHANNELS] - paired[:, OPPONENT_BOARD_CHANNELS]).mean()
        ),
        "opponent_global_mean_abs_delta": float(
            np.abs(
                own_features[:, OPPONENT_GLOBAL_SLICE] - paired_features[:, OPPONENT_GLOBAL_SLICE]
            ).mean()
        ),
        "donor_unit_action_agreement": float(
            np.mean(recipient.unit_actions[both_units] == donor.unit_actions[order][both_units])
        ),
        "donor_market_kind_agreement": float(
            np.mean(recipient.market_kinds[both_market] == donor.market_kinds[order][both_market])
        ),
    }


def ablated_inputs(
    sample: StateSample, condition: str, donor: StateSample | None, order: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Board and global features for one condition; own state is never touched."""
    if condition == "real":
        return sample.board, sample.global_features
    board = sample.board.copy()
    features = sample.global_features.copy()
    if condition == "zeroed":
        board[:, OPPONENT_BOARD_CHANNELS] = 0.0
        features[:, OPPONENT_GLOBAL_SLICE] = 0.0
        return board, features
    if condition != "transplanted":
        raise ValueError(f"unknown condition {condition!r}")
    if donor is None or order is None:
        raise ValueError("the transplant condition needs a donor sample and pairing")
    board[:, OPPONENT_BOARD_CHANNELS] = donor.board[order][:, OPPONENT_BOARD_CHANNELS]
    features[:, OPPONENT_GLOBAL_SLICE] = donor.global_features[order][:, OPPONENT_GLOBAL_SLICE]
    return board, features


# --- Forward pass and statistics -------------------------------------------


def _masked_probabilities(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Renormalized probabilities over legal choices; illegal rows stay all-zero."""
    legal = mask.any(dim=-1, keepdim=True)
    probabilities = logits.float().masked_fill(~mask, -torch.inf).softmax(dim=-1)
    return torch.where(legal, probabilities, torch.zeros_like(probabilities))


@torch.no_grad()
def head_probabilities(
    actor: torch.nn.Module,
    sample: StateSample,
    board: np.ndarray,
    features: np.ndarray,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit-head and market-kind-head distributions for every sampled state."""
    units = torch.from_numpy(sample.units)
    positions = torch.from_numpy(sample.unit_positions)
    unit_masks = torch.from_numpy(sample.unit_masks)
    kind_masks = torch.from_numpy(sample.market_kind_masks)
    unit_probabilities: list[torch.Tensor] = []
    kind_probabilities: list[torch.Tensor] = []
    for start in range(0, board.shape[0], batch_size):
        rows = slice(start, start + batch_size)
        output = actor(
            torch.from_numpy(board[rows]).float(),
            torch.from_numpy(features[rows]).float(),
            units[rows].float(),
            positions[rows],
        )
        unit_probabilities.append(_masked_probabilities(output.unit_logits, unit_masks[rows]))
        kind_probabilities.append(
            _masked_probabilities(output.market_kind_logits, kind_masks[rows])
        )
    return torch.cat(unit_probabilities), torch.cat(kind_probabilities)


def _family_masses(
    probabilities: torch.Tensor, active: torch.Tensor, families: dict[str, tuple[int, ...]]
) -> dict[str, float]:
    """Mean probability mass per family over active decisions only."""
    count = float(active.sum().item())
    return {
        name: float((probabilities[..., list(ids)].sum(dim=-1) * active).sum().item()) / count
        for name, ids in families.items()
    }


def _kl(real: torch.Tensor, other: torch.Tensor, active: torch.Tensor) -> float:
    """Mean KL(real || other) over active decisions; supports are identical."""
    floor = torch.finfo(torch.float32).tiny
    divergence = (real * (real.clamp_min(floor).log() - other.clamp_min(floor).log())).sum(dim=-1)
    return float((divergence * active).sum().item()) / float(active.sum().item())


def _selected_probability(
    probabilities: torch.Tensor, targets: torch.Tensor, active: torch.Tensor
) -> float:
    """Mean probability on the teacher's demonstrated choice, over active decisions."""
    selected = probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float((selected * active).sum().item()) / float(active.sum().item())


def _agreement(real: torch.Tensor, other: torch.Tensor, active: torch.Tensor) -> float:
    same = real.argmax(dim=-1).eq(other.argmax(dim=-1)).float()
    return float((same * active).sum().item()) / float(active.sum().item())


def head_report(
    real: torch.Tensor,
    ablations: dict[str, torch.Tensor],
    active_mask: np.ndarray,
    targets: np.ndarray,
    families: dict[str, tuple[int, ...]],
) -> dict[str, dict[str, float]]:
    if not active_mask.any():
        # Every metric below averages over active decisions; an empty head would
        # divide by zero and report a number that means nothing.
        raise ValueError("this head has no active decisions in the sample; widen the sample")
    active = torch.from_numpy(active_mask).float()
    target_tensor = torch.from_numpy(targets)
    report: dict[str, dict[str, float]] = {}
    for condition, probabilities in (("real", real), *ablations.items()):
        entry = _family_masses(probabilities, active, families)
        entry["teacher_action_probability"] = _selected_probability(
            probabilities, target_tensor, active
        )
        # Directly comparable with the clone's reported BC holdout accuracy, so
        # the poor-opponent generalization gap is readable as one number.
        entry["teacher_action_top1"] = float(
            (probabilities.argmax(dim=-1).eq(target_tensor).float() * active).sum().item()
        ) / float(active.sum().item())
        entry["kl_from_real"] = 0.0 if condition == "real" else _kl(real, probabilities, active)
        entry["argmax_agreement_with_real"] = (
            1.0 if condition == "real" else _agreement(real, probabilities, active)
        )
        report[condition] = entry
    return report


# --- Reporting --------------------------------------------------------------


def _format_table(report: dict[str, Any]) -> str:
    heads = {
        "unit": ("productive", "pass", "teacher_action_top1", "teacher_action_probability"),
        "market_kind": ("buy_seed", "sell", "stop", "teacher_action_top1"),
    }
    lines = []
    for head, keys in heads.items():
        columns = (*keys, "kl_from_real", "argmax_agreement_with_real")
        lines.append(f"{head} head")
        lines.append(
            f"  {'dataset':<24}{'condition':<14}" + "".join(f"{key[:17]:>19}" for key in columns)
        )
        for dataset in report["datasets"]:
            for condition in CONDITIONS:
                entry = dataset["heads"][head][condition]
                values = "".join(f"{entry[key]:>19.4f}" for key in columns)
                lines.append(f"  {dataset['name'][:23]:<24}{condition:<14}{values}")
    lines.append("transplant pairing (recipient vs its step-matched donor)")
    for dataset in report["datasets"]:
        pairing = dataset["transplant_pairing"]
        lines.append(
            f"  {dataset['name'][:23]:<24}donor={dataset['donor_dataset']:<24}"
            f"own board d={pairing['own_board_mean_abs_delta']:.5f} "
            f"opp board d={pairing['opponent_board_mean_abs_delta']:.5f} "
            f"opp money d={pairing['opponent_global_mean_abs_delta']:.5f} "
            f"same unit target={pairing['donor_unit_action_agreement']:.4f} "
            f"same market target={pairing['donor_market_kind_agreement']:.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    started = time.perf_counter()
    actor, payload = load_actor_artifact(args.actor, "cpu")
    orientation = checkpoint_orientation(payload)
    if orientation is not Orientation.IDENTITY:
        raise ValueError(
            f"{args.actor}: the artifact plays under {orientation.name}; this probe "
            "edits and re-encodes upright boards, so it only measures identity members"
        )
    architecture = architecture_of(actor)
    if architecture.name != CONV_ENTITY:
        raise ValueError(
            f"this probe locates the opponent slice in the {CONV_ENTITY} encoding; "
            f"the artifact is {architecture.name}"
        )
    if len(args.datasets) < 1:
        raise ValueError("--datasets needs at least one dataset directory")

    samples: list[StateSample] = []
    metadata: list[dict[str, Any]] = []
    for dataset in args.datasets:
        paths, available, manifest = _archive_paths(dataset, args.episodes_per_dataset)
        sample = load_states(paths, states_per_episode=args.states_per_episode)
        samples.append(sample)
        metadata.append(
            {
                "name": dataset.name,
                "path": str(dataset),
                "manifest_present": manifest is not None,
                "teacher": None if manifest is None else manifest["teacher"]["label"],
                "opponent": None if manifest is None else manifest["opponent"]["label"],
                "archives_available": available,
                "archives_sampled": len(sample.episodes),
                "states_sampled": int(sample.steps.shape[0]),
                # Both money features are log1p(amount)/12 (encoding._money_feature),
                # so these establish rich-vs-poor opponent empirically rather
                # than from the manifest label.
                "mean_own_money_feature": float(
                    sample.global_features[:, OWN_GLOBAL_SLICE.start].astype(np.float32).mean()
                ),
                "mean_opponent_money_feature": float(
                    sample.global_features[:, OPPONENT_GLOBAL_SLICE.start].astype(np.float32).mean()
                ),
            }
        )

    generator = np.random.default_rng(args.seed)
    report: dict[str, Any] = {
        "probe_format_version": PROBE_FORMAT_VERSION,
        "actor": {
            "path": str(args.actor),
            "sha256": file_sha256(args.actor),
            "architecture": architecture.name,
            "model_config": payload["model_config"],
        },
        "sampling": {
            "episodes_per_dataset": args.episodes_per_dataset,
            "states_per_episode": args.states_per_episode,
            "seed": args.seed,
            "rows_per_archive": samples[0].rows_per_archive,
        },
        "opponent_slices": verify_opponent_slices(args.datasets[0], samples[0].rows_per_archive),
        "families": {
            "unit": {name: [int(value) for value in ids] for name, ids in UNIT_FAMILIES.items()},
            "market_kind": {
                name: [int(value) for value in ids] for name, ids in MARKET_FAMILIES.items()
            },
        },
        "datasets": [],
    }

    for index, (sample, info) in enumerate(zip(samples, metadata, strict=True)):
        donor = samples[(index + 1) % len(samples)]
        order = donor_order(sample, donor, generator)
        conditions = {
            condition: ablated_inputs(sample, condition, donor, order) for condition in CONDITIONS
        }
        unit_probabilities: dict[str, torch.Tensor] = {}
        kind_probabilities: dict[str, torch.Tensor] = {}
        for condition, (board, features) in conditions.items():
            unit_probabilities[condition], kind_probabilities[condition] = head_probabilities(
                actor, sample, board, features, args.batch_size
            )
        entry = dict(info)
        entry["donor_dataset"] = metadata[(index + 1) % len(samples)]["name"]
        # With a single dataset the donor is the same corpus, so `transplanted`
        # permutes the opponent within one opponent style instead of crossing
        # styles; the flag keeps that readable in the record.
        entry["donor_is_same_corpus"] = len(samples) == 1
        entry["active_unit_decisions"] = int(sample.unit_active.sum())
        entry["active_market_decisions"] = int(sample.market_active.sum())
        entry["transplant_pairing"] = pairing_report(sample, donor, order)
        entry["heads"] = {
            "unit": head_report(
                unit_probabilities["real"],
                {name: unit_probabilities[name] for name in CONDITIONS[1:]},
                sample.unit_active,
                sample.unit_actions,
                UNIT_FAMILIES,
            ),
            "market_kind": head_report(
                kind_probabilities["real"],
                {name: kind_probabilities[name] for name in CONDITIONS[1:]},
                sample.market_active,
                sample.market_kinds,
                MARKET_FAMILIES,
            ),
        }
        report["datasets"].append(entry)

    report["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(_format_table(report))
    print(f"wrote {args.output} in {report['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
