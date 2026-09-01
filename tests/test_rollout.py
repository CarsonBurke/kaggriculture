from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from kaggriculture.actions import MarketKind, UnitAction
from kaggriculture.constants import DEFAULT_REWARD_GAMMA, STARTING_MONEY
from kaggriculture.encoding import pair_potential, terminal_pair_utility
from kaggriculture.model import FarmActor, ModelConfig
from kaggriculture.policy import component_logprobs
from kaggriculture.registry import CONV_ENTITY, STRUCTURED
from kaggriculture.rollout import (
    _CROP_SEED_COLUMNS,
    _PRODUCT_STOCK_COLUMNS,
    RolloutBatch,
    _builtin_agent_rows,
    _cached_compiled_forward,
    _categorical_draws,
    _native_pair_rewards,
    _state_field_specs,
    allocate_rollout_storage,
    collect_frozen_opponent_play,
    collect_frozen_opponent_play_rust,
    collect_frozen_opponents_play_rust,
    collect_mixed_play_rust,
    collect_population_play_rust,
    collect_self_play,
    collect_self_play_rust,
    concatenate_rollouts,
    merge_contiguous_rollouts,
    population_pairings,
    slice_trajectories,
)
from kaggriculture.rust_env import load_native
from kaggriculture.structured import StructuredActor, StructuredConfig, StructuredInputs


class _NearOneGenerator:
    def random(self, size):
        return np.full(size, np.nextafter(1.0, 0.0), dtype=np.float64)


def _terminal_log_ratio(own: np.ndarray, opponent: np.ndarray) -> np.ndarray:
    """Vectorized terminal log-relative bank utility used by training."""
    return np.log1p(own / STARTING_MONEY) - np.log1p(opponent / STARTING_MONEY)


def _discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    discount = float(np.float32(gamma))
    discounts = np.power(discount, np.arange(rewards.shape[1], dtype=np.float64))
    return (rewards.astype(np.float64) * discounts).sum(axis=1)


def _assert_zero_sum_reward_contract(
    rollout: RolloutBatch, *, gamma: float = DEFAULT_REWARD_GAMMA, atol: float = 1e-5
) -> None:
    # Each stored reward rounds one potential difference to binary32. Summing
    # 719 of them therefore telescopes only to binary32 accumulation accuracy.
    terminal_scores = _terminal_log_ratio(rollout.final_money, rollout.opponent_money)
    np.testing.assert_allclose(
        _discounted_returns(rollout.rewards, gamma),
        float(np.float32(gamma)) ** (rollout.horizon - 1) * terminal_scores,
        atol=atol,
    )
    assert np.isfinite(rollout.rewards).all()


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


@pytest.mark.parametrize("collector", (collect_self_play, collect_self_play_rust))
@pytest.mark.parametrize("gamma", (0.0, 1.01, float("nan")))
def test_collectors_reject_invalid_reward_gamma(collector, gamma: float) -> None:
    config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )

    with pytest.raises(ValueError, match="reward gamma must be finite"):
        collector(FarmActor(config), games=1, seed_start=1, gamma=gamma)


def test_short_self_play_rollout_preserves_discounted_terminal_utility() -> None:
    config = ModelConfig(
        cnn_width=16, cnn_blocks=1, model_dim=32, transformer_layers=3, attention_heads=4
    )
    actor = FarmActor(config)
    gamma = 0.91

    rollout = collect_self_play(
        actor,
        games=2,
        seed_start=50,
        episode_steps=8,
        deterministic=False,
        gamma=gamma,
        sampling_seed=9,
    )

    assert rollout.trajectories == 4
    assert rollout.horizon == 7
    assert rollout.state_count == 28
    assert rollout.states["board"].shape[:2] == (4, 7)
    assert rollout.unit_masks.shape[:2] == (4, 7)
    terminal_scores = _terminal_log_ratio(rollout.final_money, rollout.opponent_money)
    np.testing.assert_allclose(
        _discounted_returns(rollout.rewards, gamma),
        gamma ** (rollout.horizon - 1) * terminal_scores,
        atol=1e-6,
    )
    np.testing.assert_allclose(rollout.rewards[::2], -rollout.rewards[1::2], atol=1e-7)
    assert rollout.seats.tolist() == [0, 1, 0, 1]
    assert rollout.episode_seeds.tolist() == [50, 50, 51, 51]


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
    _assert_zero_sum_reward_contract(rollout)
    np.testing.assert_allclose(rollout.rewards[0], -rollout.rewards[1], atol=1e-7)
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
    _assert_zero_sum_reward_contract(rollout)

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
    # A single-learner wave labels every row with the one member it collected.
    assert rollout.agents.tolist() == [0, 0, 0, 0]
    np.testing.assert_array_equal(rollout.final_money[0], rollout.opponent_money[1])
    np.testing.assert_array_equal(rollout.opponent_money[0], rollout.final_money[1])

    _assert_zero_sum_reward_contract(rollout)

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
    _assert_zero_sum_reward_contract(rollout)
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
    _assert_zero_sum_reward_contract(rollout)
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


def test_native_shaped_rewards_preserve_discounted_terminal_bank_utility() -> None:
    """Every step is antisymmetric and discounted return retains terminal utility."""
    import json

    from kaggriculture.constants import MAX_MARKET_ORDERS, MAX_UNITS

    native = load_native()
    environment = native.BatchEnv(np.asarray([911], dtype=np.uint64))
    returns = np.zeros(2, dtype=np.float64)
    gamma = 0.91
    step = 0
    while True:
        unit = np.zeros((1, 2, MAX_UNITS), dtype=np.uint8)
        kinds = np.zeros((1, 2, MAX_MARKET_ORDERS), dtype=np.uint8)
        quantities = np.zeros((1, 2, MAX_MARKET_ORDERS), dtype=np.uint8)
        if step == 0:
            kinds[0, 0, 0] = MarketKind.BUY_PRODUCT_WHEAT
            quantities[0, 0, 0] = 79  # quantity bin 79 orders 80 units
        out = environment.step_factors(unit, kinds, quantities)
        rewards = _native_pair_rewards(out, gamma)
        np.testing.assert_array_equal(rewards[:, 0], -rewards[:, 1])
        if np.asarray(out["dones"]).all():
            terminal_scores = _terminal_log_ratio(
                np.asarray(out["final_money"])[:, 0],
                np.asarray(out["final_money"])[:, 1],
            )
            expected_zero = terminal_scores - out["previous_potentials"]
        else:
            expected_zero = gamma * out["potentials"] - out["previous_potentials"]
        np.testing.assert_allclose(rewards[:, 0], expected_zero, atol=1e-7)
        returns += gamma**step * rewards[0]
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
    mid_episode = pair_potential(observations[0], observations[1])
    terminal = terminal_pair_utility(observations[0], observations[1])
    assert mid_episode > terminal

    money = np.asarray(out["final_money"], dtype=np.float64)[0]
    assert money[0] > 0.0 and money[1] > 0.0
    expected_zero = _terminal_log_ratio(money[0], money[1])
    expected_terminal = np.asarray([expected_zero, -expected_zero])
    assert terminal == pytest.approx(expected_zero, abs=1e-9)
    np.testing.assert_array_equal(out["potentials"], [0.0])
    np.testing.assert_allclose(returns, gamma ** (step - 1) * expected_terminal, atol=1e-6)

    # Reset must clear the terminal potential cache; otherwise the next episode
    # starts by paying the negation of the previous game's result.
    environment.reset(np.asarray([912], dtype=np.uint64))
    reset_step = environment.step_factors(unit, kinds, quantities)
    np.testing.assert_array_equal(reset_step["previous_potentials"], [0.0])


def test_builtin_agent_rows_codes_only_the_assigned_frozen_seats() -> None:
    # Two self-play games (rows 0-3) then two league games; the learner holds
    # rows 4 and 7, so its opponents are rows 5 and 6.
    frozen_rows = np.asarray([5, 6])
    learner_rows = np.asarray([0, 1, 2, 3, 4, 7])
    lane_codes = np.asarray([0, 3], dtype=np.uint8)

    codes = _builtin_agent_rows(
        8, frozen_rows, learner_rows, np.asarray([1, 0]), lane_codes, ("frozen-0", "starter")
    )

    assert codes.dtype == np.uint8
    assert codes.tolist() == [0, 0, 0, 0, 0, 3, 0, 0]


def test_builtin_agent_rows_refuses_a_built_in_on_a_learner_seat() -> None:
    """The binding cannot see seats, so this side has to refuse the mix-up."""
    frozen_rows = np.asarray([2, 3])
    learner_rows = np.asarray([0, 1, 3])
    lane_codes = np.asarray([0, 2], dtype=np.uint8)

    with pytest.raises(ValueError, match=r"built-in lane 1 \(random\).*learner row 3"):
        _builtin_agent_rows(
            4, frozen_rows, learner_rows, np.asarray([0, 1]), lane_codes, ("frozen-0", "random")
        )


def test_native_builtin_lanes_are_played_by_the_engine_reference_agents() -> None:
    config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(config)
    opponent = FarmActor(config)

    rollout = collect_mixed_play_rust(
        actor,
        (opponent,),
        league_games=3,
        opponent_indices=np.asarray([0, 1, 2]),
        builtin_lanes=("starter", "pass"),
        seed_start=7,
        forward_mode="eager",
    )

    # Lane 2 is `pass`, which never issues a market order, so its bank has to
    # be the untouched starting money to the coin. Lane 1 is `starter`, whose
    # single-tile carrot loop is worth thousands: nothing sampled from an
    # untrained network reproduces either number by accident.
    assert rollout.opponent_money[2] == 3000.0
    assert 3000.0 < rollout.opponent_money[1] < 5000.0


def test_native_builtin_lane_needs_no_frozen_network() -> None:
    config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )

    rollout = collect_mixed_play_rust(
        FarmActor(config),
        league_games=2,
        opponent_indices=np.asarray([0, 0]),
        builtin_lanes=("pass",),
        seed_start=11,
        forward_mode="eager",
    )

    assert rollout.trajectories == 2
    assert rollout.opponent_money.tolist() == [3000.0, 3000.0]


#: Three members rather than production's four: the only full-horizon fixture
#: here pays 719 native steps, and three already scatters each member's rows
#: across the wave on both seats, which two members cannot do.
_POPULATION = 3
_POPULATION_GAMES = 6
_POPULATION_SEED_START = 310


@pytest.fixture(scope="module")
def population_wave() -> tuple[list[FarmActor], dict[str, np.ndarray], RolloutBatch]:
    """One collected population wave, shared by the properties that read it."""
    config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    # Seeded for reproducible members inside a forked CPU stream: several tests
    # in this file are sensitive to the global stream position, and a module
    # fixture runs at whichever one its first user happens to leave. The
    # collection call is inside the fork too, because the first stacked
    # ensemble in a process builds its template model from the default
    # generator. `devices=[]` keeps this off the accelerator's generators,
    # which no test here draws from.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260818)
        actors = [FarmActor(config) for _ in range(_POPULATION)]
        arena = allocate_rollout_storage(CONV_ENTITY, 2 * _POPULATION_GAMES, 719)
        rollout = collect_population_play_rust(
            actors,
            games=_POPULATION_GAMES,
            seed_start=_POPULATION_SEED_START,
            sampling_seed=4,
            storage=arena,
        )
    return actors, arena, rollout


def test_population_pairings_balance_every_ordered_pair_over_both_seats() -> None:
    pairings = population_pairings(4, 156)

    pairs, counts = np.unique(pairings, axis=0, return_counts=True)
    assert pairs.shape == (12, 2)
    assert (pairs[:, 0] != pairs[:, 1]).all()
    assert counts.tolist() == [13] * 12
    # Exactly, not in expectation: 13 games per ordered pair seats every member
    # 39 times on each side, so seat bias cancels rather than averages out.
    assert np.bincount(pairings[:, 0], minlength=4).tolist() == [39] * 4
    assert np.bincount(pairings[:, 1], minlength=4).tolist() == [39] * 4


@pytest.mark.parametrize("games", (150, 12 * 13 + 1, 0, -12))
def test_population_pairings_refuse_a_wave_size_that_cannot_balance(games: int) -> None:
    with pytest.raises(ValueError, match="positive multiple of 12"):
        population_pairings(4, games)


def test_population_pairings_refuse_a_population_with_nobody_to_play() -> None:
    with pytest.raises(ValueError, match="at least two agents"):
        population_pairings(1, 12)


def test_population_wave_refuses_a_schedule_that_starves_a_member(monkeypatch) -> None:
    """An unbalanced schedule has to fail loudly, because the fold cannot see it.

    The lanes are folded with a `view` over the rows sorted by agent, which
    succeeds for any row count divisible by the population and silently mixes
    two members into one lane when their row counts differ. That would store
    behavior log-probabilities from the wrong policy, which is indistinguishable
    from ordinary data downstream -- so the wave refuses the schedule instead.
    """
    config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    # Four rows for member 0, five for member 1 and three for member 2: still
    # twelve rows over three members, so only the counts give it away.
    starved = np.asarray([[0, 1], [0, 1], [0, 1], [0, 2], [1, 2], [2, 1]], dtype=np.int64)
    monkeypatch.setattr("kaggriculture.rollout.population_pairings", lambda *_: starved)

    with pytest.raises(ValueError, match="same row count for every member"):
        collect_population_play_rust([FarmActor(config) for _ in range(3)], games=6, seed_start=170)


def test_population_wave_stores_both_seats_in_pairing_order(population_wave) -> None:
    actors, arena, rollout = population_wave
    pairings = population_pairings(len(actors), _POPULATION_GAMES)

    # Both seats are learners, so a game yields two trajectories, game-major
    # and seat-minor against the schedule.
    assert (rollout.trajectories, rollout.horizon) == (2 * _POPULATION_GAMES, 719)
    assert rollout.state_count == 2 * _POPULATION_GAMES * 719
    for game in range(_POPULATION_GAMES):
        assert rollout.agents[2 * game] == pairings[game, 0]
        assert rollout.agents[2 * game + 1] == pairings[game, 1]
    assert rollout.agents.dtype == np.int64
    assert rollout.seats.tolist() == [0, 1] * _POPULATION_GAMES
    assert rollout.episode_seeds.tolist() == [
        seed
        for seed in range(_POPULATION_SEED_START, _POPULATION_SEED_START + _POPULATION_GAMES)
        for _ in (0, 1)
    ]
    assert (
        rollout.states["board"].__array_interface__["data"][0]
        == arena["board"].__array_interface__["data"][0]
    )


def test_population_wave_rows_replay_through_the_member_that_sampled_them(
    population_wave,
) -> None:
    """Every row's stored likelihoods must come from its own member's weights.

    This is the whole ensemble contract: one vmapped forward whose lane index
    is the agent index has to land each lane's logits back on that lane's rows,
    and the native sampler has to read that row's quantity head. A gather or
    scatter off by one lane leaves rows replaying under someone else's weights,
    which the negative control at the end shows this assertion can see.
    """
    actors, _arena, rollout = population_wave

    for row, agent in enumerate(rollout.agents.tolist()):
        _assert_stored_rows_replay_from_current_actor(
            actors[agent], slice_trajectories(rollout, row, row + 1)
        )

    other = (int(rollout.agents[0]) + 1) % len(actors)
    with pytest.raises(AssertionError):
        _assert_stored_rows_replay_from_current_actor(
            actors[other], slice_trajectories(rollout, 0, 1)
        )


def test_population_wave_rewards_are_zero_sum_within_every_game(population_wave) -> None:
    """Both learner rows receive exact opposite rewards at every transition."""
    _actors, _arena, rollout = population_wave

    np.testing.assert_array_equal(rollout.final_money[::2], rollout.opponent_money[1::2])
    np.testing.assert_array_equal(rollout.final_money[1::2], rollout.opponent_money[::2])
    expected_terminal = _terminal_log_ratio(rollout.final_money, rollout.opponent_money)
    np.testing.assert_allclose(
        _discounted_returns(rollout.rewards, DEFAULT_REWARD_GAMMA),
        DEFAULT_REWARD_GAMMA ** (rollout.horizon - 1) * expected_terminal,
        atol=2e-6,
    )
    for game in range(_POPULATION_GAMES):
        np.testing.assert_allclose(
            rollout.rewards[2 * game],
            -rollout.rewards[2 * game + 1],
            atol=1e-7,
        )


def test_slice_and_concatenation_carry_the_agent_assignment(population_wave) -> None:
    _actors, _arena, rollout = population_wave

    part = slice_trajectories(rollout, 2, 6)

    assert np.shares_memory(part.agents, rollout.agents)
    np.testing.assert_array_equal(part.agents, rollout.agents[2:6])
    combined = concatenate_rollouts(
        [slice_trajectories(rollout, 0, 4), slice_trajectories(rollout, 4, rollout.trajectories)]
    )
    np.testing.assert_array_equal(combined.agents, rollout.agents)


def test_single_learner_waves_label_every_row_as_agent_zero() -> None:
    """One learner is a population of one, so `agents` is never absent."""
    config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )

    rollout = collect_self_play(
        FarmActor(config), games=2, seed_start=150, episode_steps=3, sampling_seed=7
    )

    assert rollout.agents.tolist() == [0, 0, 0, 0]
    assert rollout.agents.dtype == np.int64
