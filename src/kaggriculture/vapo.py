"""VAPO-style masked token update for complete Kaggriculture trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from kaggriculture.model import (
    DistributionalCritic,
    FarmActor,
    distributional_value_loss,
)
from kaggriculture.policy import component_logprobs
from kaggriculture.rollout import RolloutBatch


@dataclass(frozen=True)
class VapoConfig:
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-3
    lr_warmup_steps: int = 32
    weight_decay: float = 1e-4
    epochs: int = 4
    minibatch_size: int = 2048
    clip_low: float = 0.80
    clip_high: float = 1.28
    entropy_coefficient: float = 0.01
    # Zero selects exact Monte Carlo credit. With gamma=1 this preserves the
    # telescoping potential-shaped objective exactly. Positive values expose
    # length-adaptive VAPO lambda only as an explicit ablation.
    gae_lambda_alpha: float = 0.0
    gamma: float = 1.0
    max_gradient_norm: float = 1.0
    target_kl: float = 0.08
    use_bfloat16: bool = True


@dataclass(frozen=True)
class AdvantageBatch:
    advantages: np.ndarray
    value_targets: np.ndarray
    lambdas: np.ndarray


def length_adaptive_lambda(lengths: Tensor, alpha: float = 0.0) -> Tensor:
    """Return exact-MC lambda by default, or length-adaptive VAPO lambda."""
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("GAE lambda alpha must be finite and non-negative")
    lengths = lengths.float().clamp_min(1)
    if alpha == 0.0:
        return torch.ones_like(lengths)
    horizon = torch.maximum(
        alpha * lengths,
        torch.minimum(lengths, torch.full_like(lengths, 1.0 / alpha)),
    )
    return (1.0 - 1.0 / horizon).clamp(0.0, 1.0)


def generalized_advantage_and_targets(
    rewards: Tensor,
    values: Tensor,
    valid: Tensor,
    lambdas: Tensor,
    gamma: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Masked GAE and lambda-one return targets in one reverse pass."""
    if rewards.shape != values.shape or valid.shape != values.shape:
        raise ValueError("rewards, values, and valid mask must have the same shape")
    if values.ndim != 2 or lambdas.shape != values.shape[:1]:
        raise ValueError("values must be [trajectories, time] and lambdas one per trajectory")
    if not math.isfinite(gamma) or gamma < 0.0 or gamma > 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")

    valid = valid.to(values.dtype)
    zero_column = torch.zeros_like(values[:, :1])
    next_values = torch.cat((values[:, 1:], zero_column), dim=1)
    next_valids = torch.cat((valid[:, 1:], torch.zeros_like(valid[:, :1])), dim=1)
    deltas = rewards + gamma * next_values * next_valids - values
    running_advantage = torch.zeros(values.size(0), dtype=values.dtype, device=values.device)
    running_return = torch.zeros_like(running_advantage)
    advantage_columns: list[Tensor] = []
    return_columns: list[Tensor] = []
    for step in range(values.size(1) - 1, -1, -1):
        next_valid = next_valids[:, step]
        running_advantage = (
            deltas[:, step] + gamma * lambdas * running_advantage * next_valid
        ) * valid[:, step]
        running_return = (deltas[:, step] + gamma * running_return * next_valid) * valid[:, step]
        advantage_columns.append(running_advantage)
        return_columns.append(running_return)
    advantages = torch.stack(advantage_columns[::-1], dim=1).to(values.dtype)
    return_advantages = torch.stack(return_columns[::-1], dim=1).to(values.dtype)
    return advantages, return_advantages + values * valid


def _validate_config(config: VapoConfig) -> None:
    for name, value in (
        ("actor learning rate", config.actor_learning_rate),
        ("critic learning rate", config.critic_learning_rate),
        ("max gradient norm", config.max_gradient_norm),
        ("target KL", config.target_kl),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(config.weight_decay) or config.weight_decay < 0.0:
        raise ValueError("weight decay must be finite and non-negative")
    if config.epochs < 1 or config.minibatch_size < 1:
        raise ValueError("epochs and minibatch size must be positive")
    if config.lr_warmup_steps < 0:
        raise ValueError("LR warmup steps cannot be negative")
    if not 0.0 < config.clip_low < 1.0 < config.clip_high:
        raise ValueError("clip interval must straddle one")
    if not math.isfinite(config.entropy_coefficient) or config.entropy_coefficient < 0.0:
        raise ValueError("entropy coefficient must be finite and non-negative")
    if config.gamma != 1.0:
        raise ValueError("Kaggriculture potential shaping requires undiscounted gamma=1")
    if not math.isfinite(config.gae_lambda_alpha) or config.gae_lambda_alpha < 0.0:
        raise ValueError("GAE lambda alpha must be finite and non-negative")


def prepare_advantages(rollout: RolloutBatch, config: VapoConfig) -> AdvantageBatch:
    _validate_config(config)
    rewards = torch.from_numpy(rollout.rewards).float()
    values = torch.from_numpy(rollout.old_values).float()
    valid = torch.from_numpy(rollout.valid).float()
    lengths = valid.sum(dim=1)
    lambdas = length_adaptive_lambda(lengths, config.gae_lambda_alpha)
    advantages, targets = generalized_advantage_and_targets(
        rewards, values, valid, lambdas, config.gamma
    )
    selected = advantages[valid.bool()]
    if selected.numel() == 0:
        raise ValueError("rollout contains no valid states")
    normalized = (advantages - selected.mean()) / selected.std(unbiased=False).clamp_min(1e-6)
    normalized *= valid
    return AdvantageBatch(
        advantages=normalized.numpy(),
        value_targets=targets.numpy(),
        lambdas=lambdas.numpy(),
    )


def make_optimizers(
    actor: FarmActor, critic: DistributionalCritic, config: VapoConfig
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    _validate_config(config)
    actor_device = next(actor.parameters()).device
    critic_device = next(critic.parameters()).device
    if actor_device != critic_device:
        raise ValueError("actor and critic must use the same device")
    fused = actor_device.type == "cuda"
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=config.actor_learning_rate,
        eps=1e-5,
        weight_decay=config.weight_decay,
        fused=fused,
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=config.critic_learning_rate,
        eps=1e-5,
        weight_decay=config.weight_decay,
        fused=fused,
    )
    for optimizer, base_lr in (
        (actor_optimizer, config.actor_learning_rate),
        (critic_optimizer, config.critic_learning_rate),
    ):
        for group in optimizer.param_groups:
            # Optimizer param-group metadata is checkpointed by PyTorch, so the
            # schedule resumes exactly without a separate scheduler object.
            group["base_lr"] = base_lr
            group["warmup_step"] = 0
    return actor_optimizer, critic_optimizer


def _stage_tensor(array: np.ndarray, device: torch.device) -> Tensor:
    flat = array.reshape((-1, *array.shape[2:]))
    return torch.from_numpy(flat).to(device=device)


def _batch_tensor(staged: Tensor, indices: Tensor, dtype: torch.dtype | None = None) -> Tensor:
    selected = staged.index_select(0, indices)
    return selected if dtype is None or selected.dtype == dtype else selected.to(dtype=dtype)


def _optimizer_step(
    optimizer: torch.optim.Optimizer,
    base_learning_rate: float,
    warmup_steps: int,
) -> None:
    for group in optimizer.param_groups:
        step = int(group.get("warmup_step", 0)) + 1
        group["warmup_step"] = step
        group_base_lr = float(group.get("base_lr", base_learning_rate))
        scale = min(step / warmup_steps, 1.0) if warmup_steps else 1.0
        group["lr"] = group_base_lr * scale
    optimizer.step()


def _clipped_surrogate_sums(
    new_logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    active: Tensor,
    clip_low: float,
    clip_high: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return VAPO token objective, non-negative k3 KL, and clipped count."""
    log_ratio = torch.where(active.bool(), new_logprobs.float() - old_logprobs.float(), 0.0)
    expanded_advantage = advantages.float()[:, None]
    effective_log_ratio = torch.where(
        expanded_advantage >= 0.0,
        log_ratio.clamp_max(math.log(clip_high)),
        log_ratio.clamp_min(math.log(clip_low)),
    )
    objective_sum = (effective_log_ratio.exp() * expanded_advantage * active).sum()
    approximate_kl_sum = ((torch.expm1(log_ratio) - log_ratio) * active).sum()
    clipped_sum = (
        ((log_ratio < math.log(clip_low)) | (log_ratio > math.log(clip_high))).to(active.dtype)
        * active
    ).sum()
    return objective_sum, approximate_kl_sum, clipped_sum


def _explained_variance(targets: np.ndarray, predictions: np.ndarray, valid: np.ndarray) -> float:
    selected_targets = targets[valid]
    selected_predictions = predictions[valid]
    variance = float(np.var(selected_targets))
    if variance < 1e-12:
        return 0.0
    return 1.0 - float(np.var(selected_targets - selected_predictions)) / variance


def update_vapo(
    actor: FarmActor,
    critic: DistributionalCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    config: VapoConfig,
    *,
    generator: np.random.Generator,
) -> dict[str, float | int]:
    """Replay one rollout with asymmetric, per-component clipped policy updates."""
    _validate_config(config)
    device = next(actor.parameters()).device
    if next(critic.parameters()).device != device:
        raise ValueError("actor and critic must use the same device")
    actor.train()
    critic.train()
    prepared = prepare_advantages(rollout, config)
    flat_valid = rollout.valid.reshape(-1)
    valid_indices = np.flatnonzero(flat_valid)

    # The complete rollout is reused for several PPO epochs. Stage every array
    # on the accelerator once; repeated NumPy advanced indexing otherwise makes
    # a new host copy and host-to-device transfer for every field/minibatch.
    staged = {
        "board": _stage_tensor(rollout.board, device),
        "global_features": _stage_tensor(rollout.global_features, device),
        "critic_features": _stage_tensor(rollout.critic_features, device),
        "units": _stage_tensor(rollout.units, device),
        "unit_positions": _stage_tensor(rollout.unit_positions, device),
        "unit_actions": _stage_tensor(rollout.unit_actions, device),
        "market_kinds": _stage_tensor(rollout.market_kinds, device),
        "market_quantities": _stage_tensor(rollout.market_quantities, device),
        "unit_masks": _stage_tensor(rollout.unit_masks, device),
        "market_kind_masks": _stage_tensor(rollout.market_kind_masks, device),
        "market_quantity_masks": _stage_tensor(rollout.market_quantity_masks, device),
        "unit_active": _stage_tensor(rollout.unit_active, device),
        "market_active": _stage_tensor(rollout.market_active, device),
        "market_quantity_active": _stage_tensor(rollout.market_quantity_active, device),
        "old_unit_logprobs": _stage_tensor(rollout.old_unit_logprobs, device),
        "old_market_kind_logprobs": _stage_tensor(rollout.old_market_kind_logprobs, device),
        "old_market_quantity_logprobs": _stage_tensor(rollout.old_market_quantity_logprobs, device),
        "advantages": torch.from_numpy(prepared.advantages.reshape(-1)).to(device),
        "value_targets": torch.from_numpy(prepared.value_targets.reshape(-1)).to(device),
    }
    totals = {
        key: torch.zeros((), device=device, dtype=torch.float64)
        for key in (
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "actor_gradient_norm",
            "critic_gradient_norm",
        )
    }
    total_states = 0
    total_components = 0
    updates = 0
    completed_epochs = 0
    autocast_enabled = config.use_bfloat16 and device.type == "cuda"
    stop_for_kl = False

    for _epoch in range(config.epochs):
        shuffled = generator.permutation(valid_indices)
        shuffled_device = torch.from_numpy(shuffled).to(device=device)
        for start in range(0, shuffled.size, config.minibatch_size):
            indices = shuffled_device[start : start + config.minibatch_size]
            board = _batch_tensor(staged["board"], indices, torch.float32)
            global_features = _batch_tensor(staged["global_features"], indices, torch.float32)
            critic_features = _batch_tensor(staged["critic_features"], indices, torch.float32)
            units = _batch_tensor(staged["units"], indices, torch.float32)
            positions = _batch_tensor(staged["unit_positions"], indices, torch.long)
            unit_actions = _batch_tensor(staged["unit_actions"], indices, torch.long)
            market_kinds = _batch_tensor(staged["market_kinds"], indices, torch.long)
            market_quantities = _batch_tensor(staged["market_quantities"], indices, torch.long)
            unit_masks = _batch_tensor(staged["unit_masks"], indices, torch.bool)
            kind_masks = _batch_tensor(staged["market_kind_masks"], indices, torch.bool)
            quantity_masks = _batch_tensor(staged["market_quantity_masks"], indices, torch.bool)
            unit_active = _batch_tensor(staged["unit_active"], indices, torch.float32)
            kind_active = _batch_tensor(staged["market_active"], indices, torch.float32)
            quantity_active = _batch_tensor(
                staged["market_quantity_active"], indices, torch.float32
            )
            old_unit = _batch_tensor(staged["old_unit_logprobs"], indices, torch.float32)
            old_kind = _batch_tensor(staged["old_market_kind_logprobs"], indices, torch.float32)
            old_quantity = _batch_tensor(
                staged["old_market_quantity_logprobs"], indices, torch.float32
            )
            advantages = _batch_tensor(staged["advantages"], indices, torch.float32)
            value_targets = _batch_tensor(staged["value_targets"], indices, torch.float32)
            component_count_tensor = (
                unit_active.sum() + kind_active.sum() + quantity_active.sum()
            ).clamp_min(1.0)

            actor_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                actor_output = actor(board, global_features, units, positions)
            (
                new_unit,
                new_kind,
                new_quantity,
                unit_entropy,
                kind_entropy,
                quantity_entropy,
            ) = component_logprobs(
                actor_output,
                actor.quantity_logits(
                    actor_output.market_quantity_context,
                    market_kinds,
                ),
                unit_actions,
                market_kinds,
                market_quantities,
                unit_masks,
                kind_masks,
                quantity_masks,
            )
            policy_sum = torch.zeros((), device=device)
            entropy_sum = torch.zeros((), device=device)
            kl_sum = torch.zeros((), device=device)
            clipped_sum = torch.zeros((), device=device)
            for new, old, active, entropy in (
                (new_unit, old_unit, unit_active, unit_entropy),
                (new_kind, old_kind, kind_active, kind_entropy),
                (new_quantity, old_quantity, quantity_active, quantity_entropy),
            ):
                component_objective, component_kl, component_clipped = _clipped_surrogate_sums(
                    new,
                    old,
                    advantages,
                    active,
                    config.clip_low,
                    config.clip_high,
                )
                policy_sum += component_objective
                entropy_sum += (entropy * active).sum()
                kl_sum += component_kl
                clipped_sum += component_clipped
            policy_loss = -policy_sum / component_count_tensor
            entropy_mean = entropy_sum / component_count_tensor
            actor_loss = policy_loss - config.entropy_coefficient * entropy_mean
            if not torch.isfinite(actor_loss):
                raise FloatingPointError("non-finite actor loss")
            actor_loss.backward()
            actor_gradient_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(), config.max_gradient_norm
            ).detach()
            _optimizer_step(
                actor_optimizer,
                config.actor_learning_rate,
                config.lr_warmup_steps,
            )

            critic_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                critic_logits = critic(board, critic_features)
            value_loss = distributional_value_loss(
                critic_logits, value_targets, critic.support
            ).mean()
            if not torch.isfinite(value_loss):
                raise FloatingPointError("non-finite critic loss")
            value_loss.backward()
            critic_gradient_norm = torch.nn.utils.clip_grad_norm_(
                critic.parameters(), config.max_gradient_norm
            ).detach()
            _optimizer_step(
                critic_optimizer,
                config.critic_learning_rate,
                config.lr_warmup_steps,
            )

            states = indices.numel()
            component_count = int(component_count_tensor.detach().cpu())
            batch_kl = kl_sum.detach().double() / component_count_tensor
            totals["policy_loss"] += policy_loss.detach().double() * component_count
            totals["value_loss"] += value_loss.detach().double() * states
            totals["entropy"] += entropy_mean.detach().double() * component_count
            totals["approx_kl"] += batch_kl * component_count
            totals["clip_fraction"] += clipped_sum.detach().double()
            totals["actor_gradient_norm"] += actor_gradient_norm * states
            totals["critic_gradient_norm"] += critic_gradient_norm * states
            total_components += component_count
            total_states += states
            updates += 1
            # Check after each minibatch. Waiting until the end of an epoch can
            # overshoot badly when the first from-scratch update is unstable.
            if float(batch_kl) > config.target_kl:
                stop_for_kl = True
                break
        completed_epochs += 1
        if stop_for_kl:
            break

    metrics: dict[str, float | int] = {
        "updates": updates,
        "epochs": completed_epochs,
        "states": rollout.states,
        "policy_loss": float(totals["policy_loss"] / max(1, total_components)),
        "value_loss": float(totals["value_loss"] / max(1, total_states)),
        "entropy": float(totals["entropy"] / max(1, total_components)),
        "approx_kl": float(totals["approx_kl"] / max(1, total_components)),
        "clip_fraction": float(totals["clip_fraction"] / max(1, total_components)),
        "actor_gradient_norm": float(totals["actor_gradient_norm"] / max(1, total_states)),
        "critic_gradient_norm": float(totals["critic_gradient_norm"] / max(1, total_states)),
        "advantage_mean": float(prepared.advantages[rollout.valid].mean()),
        "advantage_std": float(prepared.advantages[rollout.valid].std()),
        "value_target_mean": float(prepared.value_targets[rollout.valid].mean()),
        "value_target_std": float(prepared.value_targets[rollout.valid].std()),
        "lambda_mean": float(prepared.lambdas.mean()),
        "actor_learning_rate": float(actor_optimizer.param_groups[0]["lr"]),
        "critic_learning_rate": float(critic_optimizer.param_groups[0]["lr"]),
        "explained_variance": _explained_variance(
            prepared.value_targets, rollout.old_values, rollout.valid
        ),
    }
    return metrics
