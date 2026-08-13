from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from kaggriculture.actions import MarketKind, UnitAction
from kaggriculture.model import DistributionalCritic, FarmActor, ModelConfig
from kaggriculture.policy import component_logprobs
from kaggriculture.rollout import (
    _cached_compiled_forward,
    _categorical_draws,
    collect_frozen_opponent_play,
    collect_frozen_opponent_play_rust,
    collect_frozen_opponents_play_rust,
    collect_self_play,
    collect_self_play_rust,
    concatenate_rollouts,
)
from kaggriculture.rust_env import load_native


class _NearOneGenerator:
    def random(self, size):
        return np.full(size, np.nextafter(1.0, 0.0), dtype=np.float64)


def test_native_categorical_draw_transport_stays_strictly_below_one() -> None:
    draws = _categorical_draws(_NearOneGenerator(), rows=3)

    for component in draws:
        assert component.dtype == np.float32
        assert (component < 1.0).all()
        assert (component == np.nextafter(np.float32(1.0), np.float32(0.0))).all()


def test_short_self_play_rollout_shapes_and_telescoping() -> None:
    config = ModelConfig(width=16, residual_blocks=1, hidden=32, query_features=8)
    actor = FarmActor(config)
    critic = DistributionalCritic(config)

    rollout = collect_self_play(
        actor,
        critic,
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
    outcomes = (rollout.final_money > rollout.opponent_money).astype(float)
    outcomes -= (rollout.final_money < rollout.opponent_money).astype(float)
    assert rollout.rewards.sum(axis=1).tolist() == pytest.approx(outcomes.tolist(), abs=1e-6)
    assert rollout.seats.tolist() == [0, 1, 0, 1]
    assert rollout.episode_seeds.tolist() == [50, 50, 51, 51]
    np.testing.assert_allclose(rollout.rewards[0], -rollout.rewards[1], atol=1e-7)
    np.testing.assert_allclose(rollout.rewards[2], -rollout.rewards[3], atol=1e-7)


def test_frozen_opponent_rollout_and_concatenation() -> None:
    config = ModelConfig(width=16, residual_blocks=1, hidden=32, query_features=8)
    actor = FarmActor(config)
    critic = DistributionalCritic(config)
    opponent = FarmActor(config)
    opponent.load_state_dict(actor.state_dict())
    self_play = collect_self_play(
        actor, critic, games=1, seed_start=70, episode_steps=8, sampling_seed=1
    )
    league = collect_frozen_opponent_play(
        actor,
        critic,
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
    config = ModelConfig(width=8, residual_blocks=1, hidden=16, query_features=4)
    actor = FarmActor(config)
    critic = DistributionalCritic(config)
    first = collect_self_play(
        actor, critic, games=1, seed_start=71, episode_steps=3, sampling_seed=3
    )
    second = collect_self_play(
        actor, critic, games=1, seed_start=72, episode_steps=3, sampling_seed=4
    )
    dense_active = np.ones_like(first.unit_active)
    sparse_active = np.zeros_like(second.unit_active)
    sparse_active[..., 0] = True
    inactive_market = np.zeros_like(first.market_active)
    first = replace(
        first,
        mean_entropy=1.0,
        unit_active=dense_active,
        market_active=inactive_market,
        market_quantity_active=np.zeros_like(first.market_quantity_active),
    )
    second = replace(
        second,
        mean_entropy=3.0,
        unit_active=sparse_active,
        market_active=np.zeros_like(second.market_active),
        market_quantity_active=np.zeros_like(second.market_quantity_active),
    )

    combined = concatenate_rollouts([first, second])

    first_count = int(dense_active.sum())
    second_count = int(sparse_active.sum())
    expected = (first_count + 3.0 * second_count) / (first_count + second_count)
    assert combined.mean_entropy == pytest.approx(expected)


def test_stored_behavior_likelihoods_replay_from_identical_features() -> None:
    config = ModelConfig(width=16, residual_blocks=1, hidden=32, query_features=8)
    actor = FarmActor(config)
    critic = DistributionalCritic(config)
    rollout = collect_self_play(
        actor, critic, games=2, seed_start=110, episode_steps=8, sampling_seed=5
    )

    def flatten(values):
        return torch.from_numpy(values.reshape(-1, *values.shape[2:]))

    with torch.inference_mode():
        output = actor(
            flatten(rollout.board).float(),
            flatten(rollout.global_features).float(),
            flatten(rollout.units).float(),
            flatten(rollout.unit_positions).long(),
        )
        quantity_logits = actor.quantity_logits(
            output.market_quantity_context,
            flatten(rollout.market_kinds).long(),
        )
        unit, kind, quantity, *_ = component_logprobs(
            output,
            quantity_logits,
            flatten(rollout.unit_actions).long(),
            flatten(rollout.market_kinds).long(),
            flatten(rollout.market_quantities).long(),
            flatten(rollout.unit_masks).bool(),
            flatten(rollout.market_kind_masks).bool(),
            flatten(rollout.market_quantity_masks).bool(),
        )
        replay_values = critic.value(
            critic(flatten(rollout.board).float(), flatten(rollout.critic_features).float())
        )

    old_unit = flatten(rollout.old_unit_logprobs)
    old_kind = flatten(rollout.old_market_kind_logprobs)
    old_quantity = flatten(rollout.old_market_quantity_logprobs)
    unit_active = flatten(rollout.unit_active).bool()
    kind_active = flatten(rollout.market_active).bool()
    quantity_active = flatten(rollout.market_quantity_active).bool()
    np.testing.assert_allclose(unit[unit_active], old_unit[unit_active], atol=2e-6)
    np.testing.assert_allclose(kind[kind_active], old_kind[kind_active], atol=2e-6)
    np.testing.assert_allclose(quantity[quantity_active], old_quantity[quantity_active], atol=2e-6)
    np.testing.assert_allclose(replay_values, rollout.old_values.reshape(-1), atol=2e-6)


@pytest.mark.parametrize("collector", [collect_self_play_rust])
def test_native_self_play_rollout_is_complete_and_replayable(collector) -> None:
    config = ModelConfig(width=8, residual_blocks=1, hidden=16, query_features=4)
    actor = FarmActor(config)
    critic = DistributionalCritic(config)
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

    rollout = collector(actor, critic, games=1, seed_start=121, sampling_seed=7)

    assert rollout.trajectories == 2
    assert rollout.horizon == 719
    assert rollout.states == 1438
    outcomes = (rollout.final_money > rollout.opponent_money).astype(float)
    outcomes -= (rollout.final_money < rollout.opponent_money).astype(float)
    np.testing.assert_allclose(rollout.rewards.sum(axis=1), outcomes, atol=2e-6)
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
        replay_values = critic.value(
            critic(flatten(rollout.board).float(), flatten(rollout.critic_features).float())
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
    np.testing.assert_allclose(replay_values, rollout.old_values.reshape(-1), rtol=0.0, atol=3e-6)


def test_compiled_rollout_cache_does_not_pollute_actor_state_dict(monkeypatch) -> None:
    actor = FarmActor(ModelConfig(width=8, residual_blocks=1, hidden=16, query_features=4))

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
    config = ModelConfig(width=8, residual_blocks=1, hidden=16, query_features=4)
    actor = FarmActor(config)
    critic = DistributionalCritic(config)
    opponent = FarmActor(config)
    opponent.load_state_dict(actor.state_dict())

    rollout = collect_frozen_opponent_play_rust(
        actor,
        critic,
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
    outcomes = (rollout.final_money > rollout.opponent_money).astype(float)
    outcomes -= (rollout.final_money < rollout.opponent_money).astype(float)
    np.testing.assert_allclose(rollout.rewards.sum(axis=1), outcomes, atol=2e-6)

    # A fresh native batch exposes the same game-major/player-minor opening
    # rows. Verify that league storage selects the current seat's centralized
    # critic row, including the corresponding opponent-private features.
    initial = load_native().BatchEnv(np.asarray([130, 131], dtype=np.uint64)).encoded()
    current_rows = np.asarray([0, 3])
    np.testing.assert_array_equal(
        rollout.critic_features[:, 0], np.asarray(initial["critic_features"])[current_rows]
    )


def test_native_frozen_opponent_pool_routes_each_game_to_its_assigned_actor() -> None:
    config = ModelConfig(width=8, residual_blocks=1, hidden=16, query_features=4)
    actor = FarmActor(config)
    critic = DistributionalCritic(config)
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
    critic_state = {name: value.detach().clone() for name, value in critic.state_dict().items()}

    def fresh_models() -> tuple[FarmActor, DistributionalCritic]:
        fresh_actor = FarmActor(config)
        fresh_critic = DistributionalCritic(config)
        fresh_actor.load_state_dict(actor_state)
        fresh_critic.load_state_dict(critic_state)
        return fresh_actor, fresh_critic

    pooled = collect_frozen_opponents_play_rust(
        actor,
        critic,
        opponents,
        games=2,
        opponent_indices=np.asarray([0, 1]),
        opponent_temperatures=np.asarray([0.7, 0.9]),
        deterministic_opponents=np.asarray([True, True]),
        seed_start=140,
        sampling_seed=9,
    )
    for repeated_index in (0, 1):
        baseline_actor, baseline_critic = fresh_models()
        baseline = collect_frozen_opponents_play_rust(
            baseline_actor,
            baseline_critic,
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


@pytest.mark.parametrize("indices", [[0], [0.0, 0.0], [0, 2], [0, -1]])
def test_native_frozen_opponent_pool_rejects_invalid_assignments(indices) -> None:
    config = ModelConfig(width=8, residual_blocks=1, hidden=16, query_features=4)
    actor = FarmActor(config)
    critic = DistributionalCritic(config)
    opponent = FarmActor(config)

    with pytest.raises(ValueError):
        collect_frozen_opponents_play_rust(
            actor,
            critic,
            (opponent,),
            games=2,
            opponent_indices=indices,
            seed_start=150,
        )
