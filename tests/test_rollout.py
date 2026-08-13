from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from kaggriculture.actions import MarketKind, UnitAction
from kaggriculture.model import FarmActor, ModelConfig
from kaggriculture.policy import component_logprobs
from kaggriculture.rollout import (
    _cached_compiled_forward,
    _categorical_draws,
    allocate_rollout_storage,
    collect_frozen_opponent_play,
    collect_frozen_opponent_play_rust,
    collect_frozen_opponents_play_rust,
    collect_mixed_play_rust,
    collect_self_play,
    collect_self_play_rust,
    concatenate_rollouts,
    merge_contiguous_rollouts,
    slice_trajectories,
)
from kaggriculture.rust_env import load_native


class _NearOneGenerator:
    def random(self, size):
        return np.full(size, np.nextafter(1.0, 0.0), dtype=np.float64)


def _relative_bank_score(own: np.ndarray, opponent: np.ndarray) -> np.ndarray:
    total = own + opponent
    return np.divide(own - opponent, total, out=np.zeros_like(own), where=total != 0)


def test_native_categorical_draw_transport_stays_strictly_below_one() -> None:
    draws = _categorical_draws(_NearOneGenerator(), rows=3)

    for component in draws:
        assert component.dtype == np.float32
        assert (component < 1.0).all()
        assert (component == np.nextafter(np.float32(1.0), np.float32(0.0))).all()


@pytest.mark.parametrize("collector", (collect_self_play, collect_self_play_rust))
@pytest.mark.parametrize("temperature", (0.8, 1.2, float("nan")))
def test_on_policy_collectors_reject_nonunit_temperature(collector, temperature: float) -> None:
    config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )

    with pytest.raises(ValueError, match=r"learner temperature 1\.0"):
        collector(
            FarmActor(config),
            games=1,
            seed_start=1,
            temperature=temperature,
        )


def test_short_self_play_rollout_shapes_and_telescoping() -> None:
    config = ModelConfig(
        cnn_width=16, cnn_blocks=1, model_dim=32, transformer_layers=3, attention_heads=4
    )
    actor = FarmActor(config)

    rollout = collect_self_play(
        actor,
        games=2,
        seed_start=50,
        episode_steps=8,
        deterministic=False,
        sampling_seed=9,
    )

    assert rollout.trajectories == 4
    assert rollout.horizon == 7
    assert rollout.states == 28
    assert rollout.board.shape[:2] == (4, 7)
    assert rollout.unit_masks.shape[:2] == (4, 7)
    final_scores = _relative_bank_score(rollout.final_money, rollout.opponent_money)
    assert rollout.rewards.sum(axis=1).tolist() == pytest.approx(final_scores.tolist(), abs=1e-6)
    assert rollout.seats.tolist() == [0, 1, 0, 1]
    assert rollout.episode_seeds.tolist() == [50, 50, 51, 51]
    np.testing.assert_allclose(rollout.rewards[0], -rollout.rewards[1], atol=1e-7)
    np.testing.assert_allclose(rollout.rewards[2], -rollout.rewards[3], atol=1e-7)


def test_frozen_opponent_rollout_and_concatenation() -> None:
    config = ModelConfig(
        cnn_width=16, cnn_blocks=1, model_dim=32, transformer_layers=3, attention_heads=4
    )
    actor = FarmActor(config)
    opponent = FarmActor(config)
    opponent.load_state_dict(actor.state_dict())
    self_play = collect_self_play(actor, games=1, seed_start=70, episode_steps=8, sampling_seed=1)
    league = collect_frozen_opponent_play(
        actor,
        opponent,
        games=2,
        seed_start=80,
        episode_steps=8,
        sampling_seed=2,
    )

    combined = concatenate_rollouts([self_play, league])

    assert league.trajectories == 2
    assert league.seats.tolist() == [0, 1]
    assert combined.trajectories == 4
    assert combined.horizon == 7


def test_concatenation_weights_entropy_by_active_policy_components() -> None:
    config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(config)
    first = collect_self_play(actor, games=1, seed_start=71, episode_steps=3, sampling_seed=3)
    second = collect_self_play(actor, games=1, seed_start=72, episode_steps=3, sampling_seed=4)
    dense_active = np.ones_like(first.unit_active)
    sparse_active = np.zeros_like(second.unit_active)
    sparse_active[..., 0] = True
    inactive_market = np.zeros_like(first.market_active)
    # Per-part mean entropies of 1.0 and 3.0, expressed as per-trajectory sums
    # over each trajectory's active components.
    first = replace(
        first,
        entropy_sums=dense_active.reshape(first.trajectories, -1).sum(axis=1) * 1.0,
        unit_active=dense_active,
        market_active=inactive_market,
        market_quantity_active=np.zeros_like(first.market_quantity_active),
    )
    second = replace(
        second,
        entropy_sums=sparse_active.reshape(second.trajectories, -1).sum(axis=1) * 3.0,
        unit_active=sparse_active,
        market_active=np.zeros_like(second.market_active),
        market_quantity_active=np.zeros_like(second.market_quantity_active),
    )

    combined = concatenate_rollouts([first, second])

    first_count = int(dense_active.sum())
    second_count = int(sparse_active.sum())
    expected = (first_count + 3.0 * second_count) / (first_count + second_count)
    assert first.mean_entropy == pytest.approx(1.0)
    assert second.mean_entropy == pytest.approx(3.0)
    assert combined.mean_entropy == pytest.approx(expected)


def _flatten_states(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(values.reshape(-1, *values.shape[2:]))


def _assert_stored_rows_replay_from_current_actor(actor: FarmActor, rollout) -> None:
    """Replay every stored row through the current actor and check it matches.

    Stored behavior likelihoods must reproduce from the stored features, and
    the per-trajectory entropy sums must equal an independent recomputation
    from the replayed distributions. Both fail if any stored row was produced
    by a different policy (e.g. a frozen opponent routed into a learner row)
    or from features that differ from what the policy actually saw.
    """
    with torch.inference_mode():
        output = actor(
            _flatten_states(rollout.board).float(),
            _flatten_states(rollout.global_features).float(),
            _flatten_states(rollout.units).float(),
            _flatten_states(rollout.unit_positions).long(),
        )
        quantity_logits = actor.quantity_logits(
            output.market_quantity_context,
            _flatten_states(rollout.market_kinds).long(),
        )
        unit, kind, quantity, unit_entropy, kind_entropy, quantity_entropy = component_logprobs(
            output,
            quantity_logits,
            _flatten_states(rollout.unit_actions).long(),
            _flatten_states(rollout.market_kinds).long(),
            _flatten_states(rollout.market_quantities).long(),
            _flatten_states(rollout.unit_masks).bool(),
            _flatten_states(rollout.market_kind_masks).bool(),
            _flatten_states(rollout.market_quantity_masks).bool(),
        )

    old_unit = _flatten_states(rollout.old_unit_logprobs)
    old_kind = _flatten_states(rollout.old_market_kind_logprobs)
    old_quantity = _flatten_states(rollout.old_market_quantity_logprobs)
    unit_active = _flatten_states(rollout.unit_active).bool()
    kind_active = _flatten_states(rollout.market_active).bool()
    quantity_active = _flatten_states(rollout.market_quantity_active).bool()
    np.testing.assert_allclose(unit[unit_active], old_unit[unit_active], atol=2e-6)
    np.testing.assert_allclose(kind[kind_active], old_kind[kind_active], atol=2e-6)
    np.testing.assert_allclose(quantity[quantity_active], old_quantity[quantity_active], atol=2e-6)

    def per_trajectory(entropy: torch.Tensor, active: torch.Tensor) -> np.ndarray:
        contributions = torch.where(active, entropy.double(), torch.zeros((), dtype=torch.float64))
        return contributions.reshape(rollout.trajectories, -1).sum(dim=1).numpy()

    replayed_sums = (
        per_trajectory(unit_entropy, unit_active)
        + per_trajectory(kind_entropy, kind_active)
        + per_trajectory(quantity_entropy, quantity_active)
    )
    np.testing.assert_allclose(rollout.entropy_sums, replayed_sums, rtol=1e-5, atol=5e-3)


def test_stored_behavior_likelihoods_replay_from_identical_features() -> None:
    config = ModelConfig(
        cnn_width=16, cnn_blocks=1, model_dim=32, transformer_layers=3, attention_heads=4
    )
    actor = FarmActor(config)
    rollout = collect_self_play(actor, games=2, seed_start=110, episode_steps=8, sampling_seed=5)

    _assert_stored_rows_replay_from_current_actor(actor, rollout)


@pytest.mark.parametrize("collector", [collect_self_play_rust])
def test_native_self_play_rollout_is_complete_and_replayable(collector) -> None:
    config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(config)
    with torch.no_grad():
        # Guarantee that the compact native quantity head is exercised. The
        # conservative production prior otherwise legitimately produces
        # batches with no quantified market order at initialization.
        actor.market_kind.weight.zero_()
        actor.market_kind.bias.fill_(-12.0)
        actor.market_kind.bias[MarketKind.STOP] = -6.0
        actor.market_kind.bias[MarketKind.BUY_SEED_WHEAT] = 6.0
        actor.market_quantity_context.weight.zero_()
        actor.market_quantity_value.weight.zero_()
        actor.market_quantity_bias.fill_(-50.0)
        actor.market_quantity_bias[MarketKind.BUY_SEED_WHEAT, -1] = 50.0

    rollout = collector(actor, games=1, seed_start=121, sampling_seed=7)

    assert rollout.trajectories == 2
    assert rollout.horizon == 719
    assert rollout.states == 1438
    final_scores = _relative_bank_score(rollout.final_money, rollout.opponent_money)
    np.testing.assert_allclose(rollout.rewards.sum(axis=1), final_scores, atol=2e-6)
    assert rollout.seats.tolist() == [0, 1]
    assert np.isfinite(rollout.old_unit_logprobs).all()
    assert np.isfinite(rollout.old_market_kind_logprobs).all()
    assert np.isfinite(rollout.old_market_quantity_logprobs).all()
    assert rollout.market_quantity_active.any()
    assert rollout.market_quantity_active[:, 0, 0].all()
    np.testing.assert_array_equal(rollout.market_quantities[:, 0, 0], 99)

    def flatten(values):
        return torch.from_numpy(values.reshape(-1, *values.shape[2:]))

    with torch.inference_mode():
        output = actor(
            flatten(rollout.board).float(),
            flatten(rollout.global_features).float(),
            flatten(rollout.units).float(),
            flatten(rollout.unit_positions).long(),
        )
        unit, kind, quantity, *_ = component_logprobs(
            output,
            actor.quantity_logits(
                output.market_quantity_context,
                flatten(rollout.market_kinds).long(),
            ),
            flatten(rollout.unit_actions).long(),
            flatten(rollout.market_kinds).long(),
            flatten(rollout.market_quantities).long(),
            flatten(rollout.unit_masks).bool(),
            flatten(rollout.market_kind_masks).bool(),
            flatten(rollout.market_quantity_masks).bool(),
        )

    for replayed, behavior, active in (
        (unit, flatten(rollout.old_unit_logprobs), flatten(rollout.unit_active).bool()),
        (kind, flatten(rollout.old_market_kind_logprobs), flatten(rollout.market_active).bool()),
        (
            quantity,
            flatten(rollout.old_market_quantity_logprobs),
            flatten(rollout.market_quantity_active).bool(),
        ),
    ):
        # Rust and PyTorch both evaluate the exact selected-kind head in f32,
        # but use different reduction orders. Bound that expected roundoff as
        # well as the resulting importance-ratio error.
        np.testing.assert_allclose(replayed[active], behavior[active], rtol=0.0, atol=5e-6)
        torch.testing.assert_close(
            (replayed[active] - behavior[active]).exp(),
            torch.ones_like(replayed[active]),
            rtol=0.0,
            atol=5e-6,
        )


def test_compiled_rollout_cache_does_not_pollute_actor_state_dict(monkeypatch) -> None:
    actor = FarmActor(
        ModelConfig(
            cnn_width=8,
            cnn_blocks=1,
            model_dim=16,
            transformer_layers=3,
            attention_heads=2,
        )
    )

    class CompiledWrapper(torch.nn.Module):
        def __init__(self, wrapped: FarmActor) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, *args, **kwargs):
            return self.wrapped(*args, **kwargs)

    compiled = CompiledWrapper(actor)
    calls = []

    def fake_compile(*args, **kwargs):
        calls.append((args, kwargs))
        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    keys_before = tuple(actor.state_dict())

    assert _cached_compiled_forward(actor) is compiled
    assert _cached_compiled_forward(actor) is compiled

    assert len(calls) == 1
    _, compile_options = calls[0]
    assert compile_options == {
        "backend": "cudagraphs",
        "fullgraph": True,
        "dynamic": False,
    }
    assert tuple(actor.state_dict()) == keys_before
    assert "_kaggriculture_rollout_forward" not in actor._modules


def test_native_frozen_opponent_rollout_records_only_current_seats() -> None:
    config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(config)
    opponent = FarmActor(config)
    opponent.load_state_dict(actor.state_dict())

    rollout = collect_frozen_opponent_play_rust(
        actor,
        opponent,
        games=2,
        seed_start=130,
        sampling_seed=8,
    )

    assert rollout.trajectories == 2
    assert rollout.horizon == 719
    assert rollout.states == 1438
    assert rollout.seats.tolist() == [0, 1]
    assert rollout.episode_seeds.tolist() == [130, 131]
    final_scores = _relative_bank_score(rollout.final_money, rollout.opponent_money)
    np.testing.assert_allclose(rollout.rewards.sum(axis=1), final_scores, atol=2e-6)

    # A fresh native batch exposes the same game-major/player-minor opening
    # rows. Verify that league storage selects the current seat's centralized
    # critic row, including the corresponding opponent-private features.
    initial = load_native().BatchEnv(np.asarray([130, 131], dtype=np.uint64)).encoded()
    current_rows = np.asarray([0, 3])
    np.testing.assert_array_equal(
        rollout.critic_features[:, 0], np.asarray(initial["critic_features"])[current_rows]
    )


def test_native_frozen_opponent_pool_routes_each_game_to_its_assigned_actor() -> None:
    config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(config)
    opponents = [FarmActor(config), FarmActor(config)]
    opponents[0].load_state_dict(actor.state_dict())
    opponents[1].load_state_dict(actor.state_dict())
    with torch.no_grad():
        for opponent, action in zip(opponents, (UnitAction.PASS, UnitAction.EAST), strict=True):
            opponent.unit_head[-1].weight.zero_()
            opponent.unit_head[-1].bias.fill_(-20.0)
            opponent.unit_head[-1].bias[action] = 20.0
        for opponent, market_kind in zip(
            opponents, (MarketKind.STOP, MarketKind.HIRE), strict=True
        ):
            opponent.market_kind.weight.zero_()
            opponent.market_kind.bias.fill_(-20.0)
            opponent.market_kind.bias[market_kind] = 20.0

    actor_state = {name: value.detach().clone() for name, value in actor.state_dict().items()}

    def fresh_actor() -> FarmActor:
        restored = FarmActor(config)
        restored.load_state_dict(actor_state)
        return restored

    pooled = collect_frozen_opponents_play_rust(
        actor,
        opponents,
        games=2,
        opponent_indices=np.asarray([0, 1]),
        opponent_temperatures=np.asarray([0.7, 0.9]),
        deterministic_opponents=np.asarray([True, True]),
        seed_start=140,
        sampling_seed=9,
    )
    for repeated_index in (0, 1):
        baseline = collect_frozen_opponents_play_rust(
            fresh_actor(),
            opponents,
            games=2,
            opponent_indices=np.full(2, repeated_index),
            opponent_temperatures=np.asarray([0.7, 0.9]),
            deterministic_opponents=np.asarray([True, True]),
            seed_start=140,
            sampling_seed=9,
        )
        selected = repeated_index
        np.testing.assert_array_equal(pooled.final_money[selected], baseline.final_money[selected])
        np.testing.assert_array_equal(
            pooled.opponent_money[selected], baseline.opponent_money[selected]
        )


def test_stacked_frozen_lanes_pad_uneven_opponent_groups_exactly() -> None:
    config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(config)
    opponents = [FarmActor(config), FarmActor(config)]
    for opponent in opponents:
        opponent.load_state_dict(actor.state_dict())
    with torch.no_grad():
        for opponent, action in zip(opponents, (UnitAction.PASS, UnitAction.EAST), strict=True):
            opponent.unit_head[-1].weight.zero_()
            opponent.unit_head[-1].bias.fill_(-20.0)
            opponent.unit_head[-1].bias[action] = 20.0
        for opponent, market_kind in zip(
            opponents, (MarketKind.STOP, MarketKind.HIRE), strict=True
        ):
            opponent.market_kind.weight.zero_()
            opponent.market_kind.bias.fill_(-20.0)
            opponent.market_kind.bias[market_kind] = 20.0

    actor_state = {name: value.detach().clone() for name, value in actor.state_dict().items()}

    def fresh_actor() -> FarmActor:
        restored = FarmActor(config)
        restored.load_state_dict(actor_state)
        return restored

    # Group sizes 2 and 1 force a padded lane in the stacked frozen forward.
    padded = collect_frozen_opponents_play_rust(
        actor,
        opponents,
        games=3,
        opponent_indices=np.asarray([0, 0, 1]),
        deterministic_opponents=np.asarray([True, True]),
        seed_start=160,
        sampling_seed=11,
    )
    for repeated_index, compared_games in ((0, (0, 1)), (1, (2,))):
        baseline = collect_frozen_opponents_play_rust(
            fresh_actor(),
            opponents,
            games=3,
            opponent_indices=np.full(3, repeated_index),
            deterministic_opponents=np.asarray([True, True]),
            seed_start=160,
            sampling_seed=11,
        )
        for game in compared_games:
            np.testing.assert_array_equal(padded.final_money[game], baseline.final_money[game])
            np.testing.assert_array_equal(
                padded.opponent_money[game], baseline.opponent_money[game]
            )


@pytest.mark.parametrize("indices", [[0], [0.0, 0.0], [0, 2], [0, -1]])
def test_native_frozen_opponent_pool_rejects_invalid_assignments(indices) -> None:
    config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(config)
    opponent = FarmActor(config)

    with pytest.raises(ValueError):
        collect_frozen_opponents_play_rust(
            actor,
            (opponent,),
            games=2,
            opponent_indices=indices,
            seed_start=150,
        )


def test_arena_collection_merges_adjacent_batches_without_copying() -> None:
    config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(config)
    arena = allocate_rollout_storage(4, 719)
    first = collect_self_play_rust(
        actor,
        games=1,
        seed_start=11,
        sampling_seed=1,
        storage={name: array[:2] for name, array in arena.items()},
    )
    second = collect_self_play_rust(
        actor,
        games=1,
        seed_start=12,
        sampling_seed=2,
        storage={name: array[2:] for name, array in arena.items()},
    )

    merged = merge_contiguous_rollouts(arena, [first, second])

    assert (merged.trajectories, merged.horizon, merged.states) == (4, 719, 2876)
    assert (
        merged.board.__array_interface__["data"][0] == arena["board"].__array_interface__["data"][0]
    )
    np.testing.assert_array_equal(merged.rewards[:2], first.rewards)
    np.testing.assert_array_equal(merged.rewards[2:], second.rewards)
    np.testing.assert_array_equal(
        merged.final_money, np.concatenate([first.final_money, second.final_money])
    )
    np.testing.assert_array_equal(merged.seats, np.concatenate([first.seats, second.seats]))
    with pytest.raises(ValueError, match="not adjacent views"):
        merge_contiguous_rollouts(arena, [second, first])


def test_mixed_wave_stores_self_play_then_league_rows_in_one_arena() -> None:
    config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(config)
    # Deliberately distinct opponent weights: replaying the stored league rows
    # through the current actor below only passes if the merged wave routed
    # the learner's outputs (not an opponent's) into every stored row.
    opponents = [FarmActor(config), FarmActor(config)]

    self_play_games, league_games = 1, 2
    arena = allocate_rollout_storage(self_play_games * 2 + league_games, 719)
    rollout = collect_mixed_play_rust(
        actor,
        opponents,
        self_play_games=self_play_games,
        league_games=league_games,
        opponent_indices=np.asarray([0, 1]),
        seed_start=130,
        sampling_seed=8,
        storage=arena,
    )

    assert (rollout.trajectories, rollout.horizon, rollout.states) == (4, 719, 2876)
    assert (
        rollout.board.__array_interface__["data"][0]
        == arena["board"].__array_interface__["data"][0]
    )
    # Self-play rows come first (both seats of the same seed), league rows
    # follow with the current seat chosen by seed parity.
    assert rollout.episode_seeds.tolist() == [130, 130, 131, 132]
    assert rollout.seats.tolist() == [0, 1, 131 % 2, 132 % 2]
    np.testing.assert_array_equal(rollout.final_money[0], rollout.opponent_money[1])
    np.testing.assert_array_equal(rollout.opponent_money[0], rollout.final_money[1])

    final_scores = _relative_bank_score(rollout.final_money, rollout.opponent_money)
    np.testing.assert_allclose(rollout.rewards.sum(axis=1), final_scores, atol=2e-6)

    # The stored league rows must be the current seat's centralized critic rows
    # of the shared game-major/player-minor native batch.
    initial = load_native().BatchEnv(np.asarray([130, 131, 132], dtype=np.uint64)).encoded()
    stored_rows = np.asarray([0, 1, 2 + (131 % 2), 4 + (132 % 2)])
    np.testing.assert_array_equal(
        rollout.critic_features[:, 0], np.asarray(initial["critic_features"])[stored_rows]
    )

    # Every stored row — both self-play seats and the league current seats —
    # must replay exactly from the current actor, and the stored entropy sums
    # must match an independent recomputation from those distributions.
    assert np.isfinite(rollout.entropy_sums).all() and rollout.mean_entropy > 0.0
    _assert_stored_rows_replay_from_current_actor(actor, rollout)


def test_slice_trajectories_views_the_arena_and_validates_the_range() -> None:
    config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    rollout = collect_self_play(
        FarmActor(config), games=2, seed_start=140, episode_steps=3, sampling_seed=6
    )

    part = slice_trajectories(rollout, 1, 3)

    assert part.trajectories == 2
    assert part.horizon == rollout.horizon
    assert part.elapsed_seconds == rollout.elapsed_seconds
    assert np.shares_memory(part.board, rollout.board)
    assert np.shares_memory(part.entropy_sums, rollout.entropy_sums)
    np.testing.assert_array_equal(part.episode_seeds, rollout.episode_seeds[1:3])
    np.testing.assert_array_equal(part.rewards, rollout.rewards[1:3])
    # A view, not a copy: writes through the slice land in the parent arrays.
    part.rewards[0, 0] += 1.0
    assert rollout.rewards[1, 0] == part.rewards[0, 0]

    for start, stop in ((-1, 2), (0, 0), (2, 1), (0, rollout.trajectories + 1)):
        with pytest.raises(ValueError, match="out of range"):
            slice_trajectories(rollout, start, stop)
