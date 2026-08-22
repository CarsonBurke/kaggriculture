"""Per-member orientation behavior of the population rollout collector.

The collector flips every row's encoder output into its member's frame before
the forward, restripes the oriented movement logits back to real-action columns
for the Rust sampler, and stores the unit factors in oriented space. These
tests pin the three contracts that make the PPO replay path work unchanged:
identity is bit-for-bit the pre-orientation collector, stored factors are the
oriented images of the real sample, and non-convolutional populations are
refused rather than misread.
"""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest
import torch

from kaggriculture.actions import N_UNIT_ACTIONS
from kaggriculture.model import FarmActor, ModelConfig
from kaggriculture.orientation import (
    Orientation,
    inverse_permutation,
    movement_permutation,
    orient_unit_actions,
    orient_unit_logits,
    orient_unit_masks,
    row_orientations,
)
from kaggriculture.rollout import (
    RolloutBatch,
    collect_population_play_rust,
    slice_trajectories,
)
from kaggriculture.structured import StructuredActor, StructuredConfig

from test_rollout import _assert_stored_rows_replay_from_current_actor

_CONFIG = ModelConfig(
    cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
)
_GAMES = 2
_SEED_START = 900
_SAMPLING_SEED = 3
_WEIGHT_SEED = 20260821


def _collect_population_wave() -> tuple[list[FarmActor], RolloutBatch]:
    """One tiny deterministic two-member wave: member 1 plays mirrored in x."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_WEIGHT_SEED)
        actors = [FarmActor(_CONFIG) for _ in range(2)]
        rollout = collect_population_play_rust(
            actors,
            games=_GAMES,
            seed_start=_SEED_START,
            sampling_seed=_SAMPLING_SEED,
        )
    return actors, rollout


def _assert_batches_identical(first: RolloutBatch, second: RolloutBatch) -> None:
    """Every stored buffer of two waves agrees exactly; timing is not a buffer."""
    for field in fields(RolloutBatch):
        if field.name in ("architecture", "elapsed_seconds"):
            continue
        if field.name == "states":
            for name, array in first.states.items():
                np.testing.assert_array_equal(array, second.states[name])
        else:
            np.testing.assert_array_equal(getattr(first, field.name), getattr(second, field.name))


def test_identity_orientation_reproduces_the_pre_orientation_path(monkeypatch) -> None:
    """All-identity codes leave every collection buffer byte-identical.

    Two readings of the same fixed wave. One forces the per-row codes to
    identity, which makes every orientation helper skip its rows entirely --
    executionally the pre-orientation collector. The other keeps the real
    non-identity codes but patches each helper to a pass-through, which is what
    the collector did before orientations existed. If those two disagree, an
    all-identity population would train differently than it used to.
    """
    import kaggriculture.rollout as rollout_module

    monkeypatch.setattr(
        rollout_module,
        "row_orientations",
        lambda agents: np.zeros(np.asarray(agents).size, dtype=np.int8),
    )
    _actors, identity_run = _collect_population_wave()

    monkeypatch.setattr(rollout_module, "row_orientations", row_orientations)
    monkeypatch.setattr(rollout_module, "orient_boards", lambda board, codes: board)
    monkeypatch.setattr(
        rollout_module, "orient_unit_features", lambda units, positions, codes: None
    )
    monkeypatch.setattr(rollout_module, "orient_unit_logits", lambda logits, codes: logits)
    monkeypatch.setattr(rollout_module, "orient_unit_actions", lambda actions, codes: actions)
    monkeypatch.setattr(rollout_module, "orient_unit_masks", lambda masks, codes: masks)
    _actors, unoriented_run = _collect_population_wave()

    _assert_batches_identical(identity_run, unoriented_run)


def test_stored_factors_are_the_oriented_images_of_the_real_sample() -> None:
    """Masks, actions and replayed log-probabilities agree across label spaces.

    Synthetic stand-ins for one step's sampler output: real-space mask columns
    and real sampled actions, one row per orientation code. Storage restripes
    the columns and reindexes the actions; the replay path recomputes
    log-probabilities from oriented features, so it sees oriented logits against
    the stored oriented masks and actions. The categorical probability of the
    stored event has to equal the probability of the same real action under the
    restriped logits and the original real masks -- that equality is why the
    stored behavior log-probabilities need no remap.
    """
    rng = np.random.default_rng(7)
    rows = 4
    codes = row_orientations(np.arange(rows))
    assert sorted(int(code) for code in np.unique(codes)) == [0, 1, 2, 3]

    real_masks = rng.random((rows, 5, N_UNIT_ACTIONS)) < 0.5
    real_masks[:, :, 0] = True  # PASS is always legal, so no row masks to nothing
    real_actions = rng.integers(0, N_UNIT_ACTIONS, size=(rows, 5))
    real_masks[rows - 1, :, :] = True  # force a fully open row too

    stored_masks = orient_unit_masks(real_masks, codes)
    stored_actions = orient_unit_actions(real_actions, codes)

    oriented_logits = rng.standard_normal((rows, 5, N_UNIT_ACTIONS)).astype(np.float32)
    restriped = orient_unit_logits(oriented_logits.copy(), codes)

    def masked_logprob(logits: np.ndarray, masks: np.ndarray, actions: np.ndarray) -> np.ndarray:
        blocked = torch.finfo(torch.float32).min
        masked = torch.as_tensor(logits).masked_fill(~torch.as_tensor(masks), blocked)
        picked = torch.log_softmax(masked, dim=-1).gather(-1, torch.as_tensor(actions)[..., None])
        return picked[..., 0].numpy()

    for row, code in enumerate(codes):
        permutation = movement_permutation(Orientation(int(code)))
        inverse = inverse_permutation(permutation)
        # Stored column c describes the real action permutation[c], and the
        # stored action index executes permutation[index] in the real world.
        np.testing.assert_array_equal(stored_masks[row], real_masks[row][:, permutation])
        np.testing.assert_array_equal(permutation[stored_actions[row]], real_actions[row])
        np.testing.assert_array_equal(inverse[real_actions[row]], stored_actions[row])

        # Restriping puts each oriented logit under the real column of the
        # action it executes, so both views score one event identically.
        behavior = masked_logprob(oriented_logits[row], stored_masks[row], stored_actions[row])
        replayed = masked_logprob(restriped[row], real_masks[row], real_actions[row])
        np.testing.assert_allclose(behavior, replayed, rtol=1e-6, atol=1e-7)


def test_mirror_member_rows_replay_from_their_stored_factors() -> None:
    """The mirrored member's stored likelihoods reproduce from its own storage.

    End-to-end version of the synthetic consistency above: a real wave where
    member 1 collects every row mirrored in x. Its stored features, masks and
    actions must form one consistent oriented label space -- replaying them
    through its own weights reproduces the stored log-probabilities, while any
    real/oriented mix-up in storage would show up as a mismatch.
    """
    actors, rollout = _collect_population_wave()

    codes = row_orientations(rollout.agents)
    mirror_rows = np.flatnonzero(codes == int(Orientation.MIRROR_X))
    assert mirror_rows.tolist() != [] and set(codes[mirror_rows]) == {int(Orientation.MIRROR_X)}
    for row in mirror_rows:
        _assert_stored_rows_replay_from_current_actor(
            actors[int(rollout.agents[row])], slice_trajectories(rollout, row, row + 1)
        )


def test_structured_wave_refuses_non_identity_members() -> None:
    """The structured token encoding has no flip, so the collector refuses it.

    Fires before any simulator state exists: a structured population with
    non-identity members would otherwise silently feed unflipped boards to a
    policy trained on flipped ones.
    """
    config = StructuredConfig(
        model_dim=16,
        attention_heads=2,
        ffn_multiplier=1,
        farm_blocks=1,
        opponent_latents=2,
        latents=4,
        core_layers=1,
        quantity_rank=4,
    )
    with pytest.raises(ValueError, match="structured encoding"):
        collect_population_play_rust(
            [StructuredActor(config), StructuredActor(config)],
            games=_GAMES,
            seed_start=_SEED_START,
        )
