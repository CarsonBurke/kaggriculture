from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from kaggriculture.actions import MarketKind, UnitAction
from kaggriculture.model import FarmActor, ModelConfig
from kaggriculture.policy import component_logprobs
from kaggriculture.registry import CONV_ENTITY, STRUCTURED
from kaggriculture.rollout import (
    _CROP_SEED_COLUMNS,
    _PRODUCT_STOCK_COLUMNS,
    _cached_compiled_forward,
    _categorical_draws,
    _state_field_specs,
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
from kaggriculture.structured import StructuredActor, StructuredConfig, StructuredInputs


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
    assert rollout.state_count == 28
    assert rollout.states["board"].shape[:2] == (4, 7)
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


def _force_quantity_orders(
    actor: FarmActor | StructuredActor, *, pin_quantity_bin: bool = True
) -> None:
    """Pin the market heads to a quantified buy so the quantity path is live.

    The conservative production prior otherwise legitimately produces whole
    episodes with no quantified market order at initialization, which would
    leave the quantity component's replay assertions vacuous.

    `pin_quantity_bin` additionally collapses the quantity head onto a single
    bin, which the native comparisons need in order to name the exact bin they
    expect. Turn it off where the quantity log-probability is itself the
    measurement: a bias-only head puts the selected bin's log-probability at
    zero on both sides of a replay comparison, so the assertion degenerates to
    0 == 0, while a randomly initialized quantity head keeps real values on both
    sides -- measured -9.5 to -0.46 on the eight-step self-play fixture.
    """
    with torch.no_grad():
        actor.market_kind.weight.zero_()
        actor.market_kind.bias.fill_(-12.0)
        actor.market_kind.bias[MarketKind.STOP] = -6.0
        actor.market_kind.bias[MarketKind.BUY_SEED_WHEAT] = 6.0
        if not pin_quantity_bin:
            return
        actor.market_quantity_context.weight.zero_()
        actor.market_quantity_value.weight.zero_()
        actor.market_quantity_bias.fill_(-50.0)
        actor.market_quantity_bias[MarketKind.BUY_SEED_WHEAT, -1] = 50.0


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
            _flatten_states(rollout.states["board"]).float(),
            _flatten_states(rollout.states["global_features"]).float(),
            _flatten_states(rollout.states["units"]).float(),
            _flatten_states(rollout.states["unit_positions"]).long(),
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

    unit_active = _flatten_states(rollout.unit_active).bool()
    kind_active = _flatten_states(rollout.market_active).bool()
    quantity_active = _flatten_states(rollout.market_quantity_active).bool()
    for replayed, behavior, active in (
        (unit, _flatten_states(rollout.old_unit_logprobs), unit_active),
        (kind, _flatten_states(rollout.old_market_kind_logprobs), kind_active),
        (quantity, _flatten_states(rollout.old_market_quantity_logprobs), quantity_active),
    ):
        # An empty mask makes `assert_allclose` pass without comparing anything,
        # and that is not hypothetical here: on the eight-step fixture below the
        # quantity mask was empty at three of 24 global-RNG stream positions
        # measured, so which tests ran first decided whether this line checked
        # the quantity head at all. The structured mirror already guards it.
        assert active.any()
        np.testing.assert_allclose(replayed[active], behavior[active], atol=2e-6)

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
    # Pinned because the quantity component's coverage here is otherwise a
    # function of the global RNG stream position: over 24 positions, three left
    # `market_quantity_active` empty and the other twenty-one had exactly one
    # active component. Pinning the kind head alone puts 280 active components
    # at every one of those 24 positions, with the replay agreeing to
    # 4.8e-7..9.5e-7 against the helper's 2e-6 -- and leaving the quantity head
    # randomly initialized is what keeps that a measurement rather than a
    # comparison of two zeros (see `_force_quantity_orders`).
    _force_quantity_orders(actor, pin_quantity_bin=False)
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
    assert rollout.state_count == 1438
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
            flatten(rollout.states["board"]).float(),
            flatten(rollout.states["global_features"]).float(),
            flatten(rollout.states["units"]).float(),
            flatten(rollout.states["unit_positions"]).long(),
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
    assert rollout.state_count == 1438
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
        rollout.states["critic_features"][:, 0],
        np.asarray(initial["critic_features"])[current_rows],
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
    arena = allocate_rollout_storage(CONV_ENTITY, 4, 719)
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

    assert (merged.trajectories, merged.horizon, merged.state_count) == (4, 719, 2876)
    assert (
        merged.states["board"].__array_interface__["data"][0]
        == arena["board"].__array_interface__["data"][0]
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
    arena = allocate_rollout_storage(CONV_ENTITY, self_play_games * 2 + league_games, 719)
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

    assert (rollout.trajectories, rollout.horizon, rollout.state_count) == (4, 719, 2876)
    assert (
        rollout.states["board"].__array_interface__["data"][0]
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
        rollout.states["critic_features"][:, 0],
        np.asarray(initial["critic_features"])[stored_rows],
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
    assert np.shares_memory(part.states["board"], rollout.states["board"])
    assert np.shares_memory(part.entropy_sums, rollout.entropy_sums)
    np.testing.assert_array_equal(part.episode_seeds, rollout.episode_seeds[1:3])
    np.testing.assert_array_equal(part.rewards, rollout.rewards[1:3])
    # A view, not a copy: writes through the slice land in the parent arrays.
    part.rewards[0, 0] += 1.0
    assert rollout.rewards[1, 0] == part.rewards[0, 0]

    for start, stop in ((-1, 2), (0, 0), (2, 1), (0, rollout.trajectories + 1)):
        with pytest.raises(ValueError, match="out of range"):
            slice_trajectories(rollout, start, stop)


def _small_structured_config() -> StructuredConfig:
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


def _structured_replay_inputs(rollout) -> StructuredInputs:
    states = rollout.states
    return StructuredInputs(
        tile_categorical=_flatten_states(states["tile_categorical"]).long(),
        tile_continuous=_flatten_states(states["tile_continuous"]).float(),
        unit_categorical=_flatten_states(states["unit_categorical"]).long(),
        unit_continuous=_flatten_states(states["unit_continuous"]).float(),
        unit_active=_flatten_states(rollout.unit_active).bool(),
        unit_tile_gather=_flatten_states(states["unit_tile_gather"]).long(),
        unit_tile_gather_valid=_flatten_states(states["unit_tile_gather_valid"]).bool(),
        products=_flatten_states(states["products"]).float(),
        crops=_flatten_states(states["crops"]).float(),
        farms=_flatten_states(states["farms"]).float(),
        town=_flatten_states(states["town"]).float(),
    )


def _assert_structured_rows_replay_from_current_actor(
    actor: StructuredActor, rollout, atol: float
) -> None:
    """Structured mirror of the convolutional replay-and-entropy audit."""
    with torch.inference_mode():
        output = actor(_structured_replay_inputs(rollout))
        market_kinds = _flatten_states(rollout.market_kinds).long()
        unit, kind, quantity, unit_entropy, kind_entropy, quantity_entropy = component_logprobs(
            output,
            actor.quantity_logits(output.market_quantity_context, market_kinds),
            _flatten_states(rollout.unit_actions).long(),
            market_kinds,
            _flatten_states(rollout.market_quantities).long(),
            _flatten_states(rollout.unit_masks).bool(),
            _flatten_states(rollout.market_kind_masks).bool(),
            _flatten_states(rollout.market_quantity_masks).bool(),
        )

    unit_active = _flatten_states(rollout.unit_active).bool()
    kind_active = _flatten_states(rollout.market_active).bool()
    quantity_active = _flatten_states(rollout.market_quantity_active).bool()
    for replayed, behavior, active in (
        (unit, _flatten_states(rollout.old_unit_logprobs), unit_active),
        (kind, _flatten_states(rollout.old_market_kind_logprobs), kind_active),
        (quantity, _flatten_states(rollout.old_market_quantity_logprobs), quantity_active),
    ):
        assert active.any()
        np.testing.assert_allclose(replayed[active], behavior[active], rtol=0.0, atol=atol)

    def per_trajectory(entropy: torch.Tensor, active: torch.Tensor) -> np.ndarray:
        contributions = torch.where(active, entropy.double(), torch.zeros((), dtype=torch.float64))
        return contributions.reshape(rollout.trajectories, -1).sum(dim=1).numpy()

    replayed_sums = (
        per_trajectory(unit_entropy, unit_active)
        + per_trajectory(kind_entropy, kind_active)
        + per_trajectory(quantity_entropy, quantity_active)
    )
    np.testing.assert_allclose(rollout.entropy_sums, replayed_sums, rtol=1e-5, atol=5e-3)


def test_structured_self_play_rollout_replays_from_stored_states() -> None:
    actor = StructuredActor(_small_structured_config())
    _force_quantity_orders(actor)

    rollout = collect_self_play(actor, games=1, seed_start=210, episode_steps=8, sampling_seed=13)

    assert rollout.architecture == STRUCTURED
    assert (rollout.trajectories, rollout.horizon, rollout.state_count) == (2, 7, 14)
    assert set(rollout.states) == set(_state_field_specs(STRUCTURED))
    for name, (shape, dtype) in _state_field_specs(STRUCTURED).items():
        assert rollout.states[name].shape == (2, 7, *shape)
        assert rollout.states[name].dtype == dtype
    final_scores = _relative_bank_score(rollout.final_money, rollout.opponent_money)
    np.testing.assert_allclose(rollout.rewards.sum(axis=1), final_scores, atol=2e-6)
    np.testing.assert_allclose(rollout.rewards[0], -rollout.rewards[1], atol=1e-7)
    _assert_structured_rows_replay_from_current_actor(actor, rollout, atol=2e-6)


def test_native_structured_sampler_and_encoder_agree_on_unit_activity() -> None:
    """The sampled factor mask and the attention mask must be one predicate.

    The Rust sampler and the Rust structured encoder each compute "unit slot
    is an existing unit" independently; the rollout stores the sampler's
    answer and reuses it as StructuredInputs.unit_active in the update path.
    Replaying the stored actions through a fresh environment compares the
    encoder's per-step answer against the stored sampler mask, so any future
    asymmetric edit to either predicate fails here instead of silently
    biasing the update's attention masking off-policy.
    """
    actor = StructuredActor(_small_structured_config())
    with torch.no_grad():
        # Force hires so unit activity actually grows past the opening farmer;
        # otherwise the identity below is only tested on constant masks.
        actor.market_kind.weight.zero_()
        actor.market_kind.bias.fill_(-20.0)
        actor.market_kind.bias[MarketKind.HIRE] = 20.0

    rollout = collect_self_play_rust(actor, games=1, seed_start=217, sampling_seed=19)

    environment = load_native().BatchEnv(np.asarray([217], dtype=np.uint64))
    for step in range(rollout.horizon):
        encoded = environment.structured()
        np.testing.assert_array_equal(
            rollout.unit_active[:, step], np.asarray(encoded["unit_active"])
        )
        environment.step_factors(
            rollout.unit_actions[:, step].astype(np.uint8)[None],
            rollout.market_kinds[:, step].astype(np.uint8)[None],
            rollout.market_quantities[:, step].astype(np.uint8)[None],
        )
    assert rollout.unit_active.sum() > rollout.trajectories * rollout.horizon


def test_structured_mixed_wave_stores_and_replays_in_one_arena() -> None:
    config = _small_structured_config()
    actor = StructuredActor(config)
    _force_quantity_orders(actor)
    # Deliberately distinct opponent weights: the replay below only passes if
    # the merged wave routed the learner's outputs into every stored row.
    opponents = [StructuredActor(config), StructuredActor(config)]

    self_play_games, league_games = 1, 2
    arena = allocate_rollout_storage(STRUCTURED, self_play_games * 2 + league_games, 719)
    rollout = collect_mixed_play_rust(
        actor,
        opponents,
        self_play_games=self_play_games,
        league_games=league_games,
        opponent_indices=np.asarray([0, 1]),
        seed_start=230,
        sampling_seed=21,
        storage=arena,
    )

    assert rollout.architecture == STRUCTURED
    assert (rollout.trajectories, rollout.horizon, rollout.state_count) == (4, 719, 2876)
    assert (
        rollout.states["tile_continuous"].__array_interface__["data"][0]
        == arena["tile_continuous"].__array_interface__["data"][0]
    )
    assert rollout.episode_seeds.tolist() == [230, 230, 231, 232]
    assert rollout.seats.tolist() == [0, 1, 231 % 2, 232 % 2]
    final_scores = _relative_bank_score(rollout.final_money, rollout.opponent_money)
    np.testing.assert_allclose(rollout.rewards.sum(axis=1), final_scores, atol=2e-6)
    assert np.isfinite(rollout.old_unit_logprobs).all()
    assert np.isfinite(rollout.old_market_kind_logprobs).all()
    assert np.isfinite(rollout.old_market_quantity_logprobs).all()
    assert rollout.market_quantity_active[:, 0, 0].all()

    # The centralized-critic extras must be the paired seat's own view of the
    # shared game-major/player-minor native batch, with the private economy
    # columns sliced from the paired seat's token buffers.
    initial = load_native().BatchEnv(np.asarray([230, 231, 232], dtype=np.uint64)).structured()
    stored_rows = np.asarray([0, 1, 2 + (231 % 2), 4 + (232 % 2)])
    pair_rows = stored_rows ^ 1
    np.testing.assert_array_equal(
        rollout.states["unit_categorical"][:, 0],
        np.asarray(initial["unit_categorical"])[stored_rows],
    )
    np.testing.assert_array_equal(
        rollout.states["opponent_unit_categorical"][:, 0],
        np.asarray(initial["unit_categorical"])[pair_rows],
    )
    np.testing.assert_array_equal(
        rollout.states["opponent_unit_active"][:, 0],
        np.asarray(initial["unit_active"])[pair_rows],
    )
    np.testing.assert_array_equal(
        rollout.states["critic_products"][:, 0],
        np.asarray(initial["products"])[pair_rows][:, :, _PRODUCT_STOCK_COLUMNS],
    )
    np.testing.assert_array_equal(
        rollout.states["critic_crops"][:, 0],
        np.asarray(initial["crops"])[pair_rows][:, :, _CROP_SEED_COLUMNS],
    )

    assert np.isfinite(rollout.entropy_sums).all() and rollout.mean_entropy > 0.0
    _assert_structured_rows_replay_from_current_actor(actor, rollout, atol=5e-6)


def test_native_terminal_reward_scores_bank_while_holdings_stay_liquid() -> None:
    """The done-step potential must switch from liquidation value to bank.

    Forces player zero to end the episode holding a large unsold shed, so the
    terminal branch is observable: with products still held, relative bank and
    relative liquidation value differ, and the telescoped shaped return must
    equal the exact relative final bank rather than the liquidation score.
    Random-policy episodes cannot guard this — they end bankrupt and empty,
    making the terminal reward zero either way.
    """
    import json

    from kaggriculture.constants import MAX_MARKET_ORDERS, MAX_UNITS
    from kaggriculture.encoding import pair_potential, terminal_pair_potential

    native = load_native()
    environment = native.BatchEnv(np.asarray([911], dtype=np.uint64))
    telescoped = 0.0
    step = 0
    while True:
        unit = np.zeros((1, 2, MAX_UNITS), dtype=np.uint8)
        kinds = np.zeros((1, 2, MAX_MARKET_ORDERS), dtype=np.uint8)
        quantities = np.zeros((1, 2, MAX_MARKET_ORDERS), dtype=np.uint8)
        if step == 0:
            kinds[0, 0, 0] = MarketKind.BUY_PRODUCT_WHEAT
            quantities[0, 0, 0] = 79  # quantity bin 79 orders 80 units
        out = environment.step_factors(unit, kinds, quantities)
        telescoped += float(np.asarray(out["shaped_rewards"])[0, 0])
        step += 1
        if np.asarray(out["dones"]).all():
            break

    frozen = json.loads(environment.snapshot_json(0))
    observations = [
        {
            "player": player,
            "farms": frozen["farms"],
            "private": frozen["privates"][player],
            "market": frozen["market"],
        }
        for player in range(2)
    ]
    assert observations[0]["private"]["shed"]["WHEAT"] == 80
    terminal = terminal_pair_potential(observations[0], observations[1])
    mid_episode = pair_potential(observations[0], observations[1])
    assert abs(terminal - mid_episode) > 0.01

    money = np.asarray(out["final_money"], dtype=np.float64)[0]
    assert money[0] > 0.0 and money[1] > 0.0
    relative_bank = (money[0] - money[1]) / (money[0] + money[1])
    assert terminal == pytest.approx(relative_bank, abs=1e-9)
    # Native terminal potential matches the python mirror bank-only score.
    assert float(np.asarray(out["potentials"])[0]) == pytest.approx(terminal, abs=1e-7)
    # The shaped return telescopes to the exact relative final bank, not to
    # the liquidation score of the unsold shed.
    assert telescoped == pytest.approx(relative_bank, abs=1e-5)
