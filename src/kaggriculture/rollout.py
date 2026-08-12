"""Synchronous self-play rollout collection against the official simulator."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from kaggle_environments import make

from kaggriculture.encoding import pair_potential, shaped_pair_reward
from kaggriculture.model import DistributionalCritic, FarmActor
from kaggriculture.policy import PolicyStep, act_batch


@dataclass(frozen=True)
class RolloutBatch:
    board: np.ndarray
    global_features: np.ndarray
    critic_features: np.ndarray
    units: np.ndarray
    unit_positions: np.ndarray
    unit_actions: np.ndarray
    market_kinds: np.ndarray
    market_quantities: np.ndarray
    unit_masks: np.ndarray
    market_kind_masks: np.ndarray
    market_quantity_masks: np.ndarray
    unit_active: np.ndarray
    market_active: np.ndarray
    market_quantity_active: np.ndarray
    old_unit_logprobs: np.ndarray
    old_market_kind_logprobs: np.ndarray
    old_market_quantity_logprobs: np.ndarray
    old_values: np.ndarray
    rewards: np.ndarray
    valid: np.ndarray
    episode_seeds: np.ndarray
    final_money: np.ndarray
    opponent_money: np.ndarray
    seats: np.ndarray
    mean_entropy: float
    elapsed_seconds: float

    @property
    def trajectories(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.rewards.shape[1])

    @property
    def states(self) -> int:
        return int(self.valid.sum())


def _trajectory_first(values: list[np.ndarray], dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.stack(values, axis=1)
    if dtype is not None:
        result = result.astype(dtype, copy=False)
    return result


def _observations(states: list[list[Any]]) -> list[dict[str, Any]]:
    return [agent.observation for state in states for agent in state]


def _opponent_privates(states: list[list[Any]]) -> list[dict[str, Any]]:
    out = []
    for state in states:
        out.extend((state[1].observation["private"], state[0].observation["private"]))
    return out


def _new_fields() -> dict[str, list[np.ndarray]]:
    return {
        name: []
        for name in (
            "board",
            "global_features",
            "critic_features",
            "units",
            "unit_positions",
            "unit_actions",
            "market_kinds",
            "market_quantities",
            "unit_masks",
            "market_kind_masks",
            "market_quantity_masks",
            "unit_active",
            "market_active",
            "market_quantity_active",
            "old_unit_logprobs",
            "old_market_kind_logprobs",
            "old_market_quantity_logprobs",
            "old_values",
            "rewards",
            "valid",
        )
    }


def _record_policy_step(fields: dict[str, list[np.ndarray]], policy_step: PolicyStep) -> None:
    factors = policy_step.factors
    fields["board"].append(np.stack([row.board for row in policy_step.encoded]).astype(np.float16))
    fields["global_features"].append(
        np.stack([row.global_features for row in policy_step.encoded]).astype(np.float16)
    )
    fields["critic_features"].append(
        np.stack([row.critic_features for row in policy_step.encoded]).astype(np.float16)
    )
    fields["units"].append(np.stack([row.units for row in policy_step.encoded]).astype(np.float16))
    fields["unit_positions"].append(
        np.stack([row.unit_positions for row in policy_step.encoded]).astype(np.int8)
    )
    fields["unit_actions"].append(factors.unit_actions.astype(np.int8))
    fields["market_kinds"].append(factors.market_kinds.astype(np.int8))
    fields["market_quantities"].append(factors.market_quantities.astype(np.int8))
    fields["unit_masks"].append(factors.unit_masks)
    fields["market_kind_masks"].append(factors.market_kind_masks)
    fields["market_quantity_masks"].append(factors.market_quantity_masks)
    fields["unit_active"].append(factors.unit_active)
    fields["market_active"].append(factors.market_active)
    fields["market_quantity_active"].append(factors.market_quantity_active)
    fields["old_unit_logprobs"].append(factors.unit_logprobs)
    fields["old_market_kind_logprobs"].append(factors.market_kind_logprobs)
    fields["old_market_quantity_logprobs"].append(factors.market_quantity_logprobs)
    fields["old_values"].append(factors.values)


def _finish_rollout(
    fields: dict[str, list[np.ndarray]],
    *,
    episode_seeds: np.ndarray,
    final_money: np.ndarray,
    opponent_money: np.ndarray,
    seats: np.ndarray,
    entropies: list[float],
    started: float,
) -> RolloutBatch:
    return RolloutBatch(
        board=_trajectory_first(fields["board"]),
        global_features=_trajectory_first(fields["global_features"]),
        critic_features=_trajectory_first(fields["critic_features"]),
        units=_trajectory_first(fields["units"]),
        unit_positions=_trajectory_first(fields["unit_positions"]),
        unit_actions=_trajectory_first(fields["unit_actions"]),
        market_kinds=_trajectory_first(fields["market_kinds"]),
        market_quantities=_trajectory_first(fields["market_quantities"]),
        unit_masks=_trajectory_first(fields["unit_masks"]),
        market_kind_masks=_trajectory_first(fields["market_kind_masks"]),
        market_quantity_masks=_trajectory_first(fields["market_quantity_masks"]),
        unit_active=_trajectory_first(fields["unit_active"]),
        market_active=_trajectory_first(fields["market_active"]),
        market_quantity_active=_trajectory_first(fields["market_quantity_active"]),
        old_unit_logprobs=_trajectory_first(fields["old_unit_logprobs"]),
        old_market_kind_logprobs=_trajectory_first(fields["old_market_kind_logprobs"]),
        old_market_quantity_logprobs=_trajectory_first(fields["old_market_quantity_logprobs"]),
        old_values=_trajectory_first(fields["old_values"]),
        rewards=_trajectory_first(fields["rewards"]),
        valid=_trajectory_first(fields["valid"]),
        episode_seeds=episode_seeds,
        final_money=final_money,
        opponent_money=opponent_money,
        seats=seats,
        mean_entropy=float(np.mean(entropies)),
        elapsed_seconds=time.perf_counter() - started,
    )


def collect_self_play(
    actor: FarmActor,
    critic: DistributionalCritic,
    *,
    games: int,
    seed_start: int,
    episode_steps: int = 720,
    deterministic: bool = False,
    temperature: float = 1.0,
    sampling_seed: int = 0,
) -> RolloutBatch:
    """Collect both valid on-policy trajectories from every self-play game."""
    if games < 1:
        raise ValueError("games must be positive")
    if episode_steps < 2:
        raise ValueError("episode_steps must be at least two")
    actor.eval()
    critic.eval()
    generator = np.random.default_rng(sampling_seed)
    environments = [
        make(
            "kaggriculture",
            configuration={"episodeSteps": episode_steps, "seed": seed_start + index},
            debug=False,
        )
        for index in range(games)
    ]
    states = [environment.reset(2) for environment in environments]
    potentials = [pair_potential(state[0].observation, state[1].observation) for state in states]
    fields = _new_fields()
    trajectories = games * 2
    final_money = np.zeros(trajectories, dtype=np.float32)
    opponent_money = np.zeros(trajectories, dtype=np.float32)
    entropies = []
    started = time.perf_counter()

    for _ in range(episode_steps):
        if all(environment.done for environment in environments):
            break
        if any(environment.done for environment in environments):
            raise RuntimeError("synchronous environments terminated at different horizons")
        policy_step = act_batch(
            actor,
            critic,
            _observations(states),
            _opponent_privates(states),
            deterministic=deterministic,
            temperature=temperature,
            generator=generator,
        )
        _record_policy_step(fields, policy_step)
        entropies.append(policy_step.factors.entropy)

        next_states = []
        step_rewards = np.zeros(trajectories, dtype=np.float32)
        for game, environment in enumerate(environments):
            offset = game * 2
            next_state = environment.step(policy_step.actions[offset : offset + 2])
            next_states.append(next_state)
            if any(agent.status == "ERROR" for agent in next_state):
                raise RuntimeError(f"agent error in self-play seed {seed_start + game}")
            if environment.done:
                farms = next_state[0].observation["farms"]
                money = np.asarray(
                    [float(farms[0]["money"]), float(farms[1]["money"])], dtype=np.float32
                )
                final_money[offset : offset + 2] = money
                opponent_money[offset : offset + 2] = money[::-1]
                pair_rewards = shaped_pair_reward(potentials[game], 0.0, float(money[0] - money[1]))
            else:
                next_potential = pair_potential(
                    next_state[0].observation, next_state[1].observation
                )
                pair_rewards = shaped_pair_reward(potentials[game], next_potential)
                potentials[game] = next_potential
            step_rewards[offset : offset + 2] = pair_rewards
        fields["rewards"].append(step_rewards)
        fields["valid"].append(np.ones(trajectories, dtype=np.bool_))
        states = next_states
    else:
        raise RuntimeError("self-play rollout exceeded the configured episode horizon")

    if not all(environment.done for environment in environments):
        raise RuntimeError("self-play rollout ended before all environments reached DONE")
    return _finish_rollout(
        fields,
        episode_seeds=np.repeat(
            np.arange(seed_start, seed_start + games, dtype=np.int64), repeats=2
        ),
        final_money=final_money,
        opponent_money=opponent_money,
        seats=np.tile(np.asarray([0, 1], dtype=np.int8), games),
        entropies=entropies,
        started=started,
    )


def collect_frozen_opponent_play(
    actor: FarmActor,
    critic: DistributionalCritic,
    opponent: FarmActor,
    *,
    games: int,
    seed_start: int,
    episode_steps: int = 720,
    temperature: float = 1.0,
    opponent_temperature: float = 0.8,
    deterministic_opponent: bool = False,
    deterministic: bool = False,
    sampling_seed: int = 0,
) -> RolloutBatch:
    """Collect one current-policy trajectory per game against a frozen snapshot."""
    if games < 1:
        raise ValueError("games must be positive")
    if episode_steps < 2:
        raise ValueError("episode_steps must be at least two")
    actor.eval()
    critic.eval()
    opponent.eval()
    device = next(actor.parameters()).device
    if next(opponent.parameters()).device != device:
        raise ValueError("current and frozen policies must use the same device")
    current_generator = np.random.default_rng(sampling_seed)
    opponent_generator = np.random.default_rng(sampling_seed ^ 0x5EED_1EAF)
    environments = [
        make(
            "kaggriculture",
            configuration={"episodeSteps": episode_steps, "seed": seed_start + index},
            debug=False,
        )
        for index in range(games)
    ]
    states = [environment.reset(2) for environment in environments]
    seats = np.asarray([(seed_start + index) % 2 for index in range(games)], dtype=np.int8)
    potentials = [pair_potential(state[0].observation, state[1].observation) for state in states]
    fields = _new_fields()
    final_money = np.zeros(games, dtype=np.float32)
    opponent_money = np.zeros(games, dtype=np.float32)
    entropies = []
    started = time.perf_counter()

    for _ in range(episode_steps):
        if all(environment.done for environment in environments):
            break
        if any(environment.done for environment in environments):
            raise RuntimeError("synchronous environments terminated at different horizons")
        current_observations = [
            state[int(seat)].observation for state, seat in zip(states, seats, strict=True)
        ]
        frozen_observations = [
            state[1 - int(seat)].observation for state, seat in zip(states, seats, strict=True)
        ]
        current_step = act_batch(
            actor,
            critic,
            current_observations,
            [observation["private"] for observation in frozen_observations],
            deterministic=deterministic,
            temperature=temperature,
            generator=current_generator,
        )
        frozen_step = act_batch(
            opponent,
            None,
            frozen_observations,
            deterministic=deterministic_opponent,
            temperature=opponent_temperature,
            generator=opponent_generator,
        )
        _record_policy_step(fields, current_step)
        entropies.append(current_step.factors.entropy)

        next_states = []
        step_rewards = np.zeros(games, dtype=np.float32)
        for game, (environment, seat) in enumerate(zip(environments, seats, strict=True)):
            actions = [None, None]
            actions[int(seat)] = current_step.actions[game]
            actions[1 - int(seat)] = frozen_step.actions[game]
            next_state = environment.step(actions)
            next_states.append(next_state)
            if any(agent.status == "ERROR" for agent in next_state):
                raise RuntimeError(f"agent error in league seed {seed_start + game}")
            if environment.done:
                farms = next_state[0].observation["farms"]
                player_money = (float(farms[0]["money"]), float(farms[1]["money"]))
                final_money[game] = player_money[int(seat)]
                opponent_money[game] = player_money[1 - int(seat)]
                pair_rewards = shaped_pair_reward(
                    potentials[game], 0.0, player_money[0] - player_money[1]
                )
            else:
                next_potential = pair_potential(
                    next_state[0].observation, next_state[1].observation
                )
                pair_rewards = shaped_pair_reward(potentials[game], next_potential)
                potentials[game] = next_potential
            step_rewards[game] = pair_rewards[int(seat)]
        fields["rewards"].append(step_rewards)
        fields["valid"].append(np.ones(games, dtype=np.bool_))
        states = next_states
    else:
        raise RuntimeError("league rollout exceeded the configured episode horizon")

    if not all(environment.done for environment in environments):
        raise RuntimeError("league rollout ended before all environments reached DONE")
    return _finish_rollout(
        fields,
        episode_seeds=np.arange(seed_start, seed_start + games, dtype=np.int64),
        final_money=final_money,
        opponent_money=opponent_money,
        seats=seats,
        entropies=entropies,
        started=started,
    )


def concatenate_rollouts(batches: list[RolloutBatch]) -> RolloutBatch:
    """Concatenate compatible current-policy trajectories from several match sources."""
    if not batches:
        raise ValueError("at least one rollout batch is required")
    if len(batches) == 1:
        return batches[0]
    if len({batch.horizon for batch in batches}) != 1:
        raise ValueError("rollout horizons must match")
    array_fields = (
        "board",
        "global_features",
        "critic_features",
        "units",
        "unit_positions",
        "unit_actions",
        "market_kinds",
        "market_quantities",
        "unit_masks",
        "market_kind_masks",
        "market_quantity_masks",
        "unit_active",
        "market_active",
        "market_quantity_active",
        "old_unit_logprobs",
        "old_market_kind_logprobs",
        "old_market_quantity_logprobs",
        "old_values",
        "rewards",
        "valid",
        "episode_seeds",
        "final_money",
        "opponent_money",
        "seats",
    )
    combined = {
        field: np.concatenate([getattr(batch, field) for batch in batches], axis=0)
        for field in array_fields
    }
    total_states = sum(batch.states for batch in batches)
    mean_entropy = sum(batch.mean_entropy * batch.states for batch in batches) / max(
        1, total_states
    )
    return RolloutBatch(
        **combined,
        mean_entropy=mean_entropy,
        elapsed_seconds=sum(batch.elapsed_seconds for batch in batches),
    )
