"""VAPO masked-token update with decoupled actor GAE and critic returns."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from kaggriculture.constants import EPISODE_STEPS
from kaggriculture.model import (
    DistributionalCritic,
    FarmActor,
    distributional_value_loss,
)
from kaggriculture.policy import component_logprobs, component_selected_logprobs
from kaggriculture.rollout import RolloutBatch

VAPO_GAE_ALPHA = 0.05
COMPETITION_ACTION_STEPS = EPISODE_STEPS - 1
DEFAULT_ACTOR_GAE_LAMBDA = 1.0 - 1.0 / (VAPO_GAE_ALPHA * COMPETITION_ACTION_STEPS)
CRITIC_GAE_LAMBDA = 1.0

# Numerics gates shared by the calibration benchmark, the training launcher's
# expected configuration, and the production training loop. The sampling-path
# versus update-replay ratio divergence is irreducible bf16 noise: measured
# worst cases are 2.35e-2 over 230k production states at initialization and
# 1.9e-2 across seed, rollout-size, and sharpened-head sweeps, while real
# staging or precision bugs present orders of magnitude larger, so 5e-2 keeps
# roughly 2x headroom while spending only a fraction of the [0.8, 1.28] clip
# band. The first-minibatch KL at unchanged weights measures ~7e-8 under
# compile with bf16; 1e-4 bounds the ratio-at-one construction with margin.
MAX_UPDATE_REPLAY_RATIO_ERROR = 5e-2
MAX_FIRST_MINIBATCH_KL = 1e-4


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
    # VAPO's lambda_policy = 1 - 1 / (alpha * length), with alpha=0.05 and the
    # competition's fixed 719-action horizon. The critic is deliberately
    # decoupled below and always learns from lambda-one Monte Carlo returns.
    actor_gae_lambda: float = DEFAULT_ACTOR_GAE_LAMBDA
    gamma: float = 1.0
    max_gradient_norm: float = 1.0
    target_kl: float = 0.08
    # BF16 autocast for both update-path forwards. The actor's importance
    # ratio starts at one because `replay_behavior_logprobs` recomputes the
    # behavior side through the update path's forward at the same precision;
    # log_softmax stays fp32 under autocast either way.
    use_bfloat16: bool = True
    # Compile the update-path forward/backward with Inductor. Fusion collapses
    # the launch-bound logprob/surrogate math into a few large kernels while
    # keeping memory eager-like, unlike CUDA-graph capture whose per-minibatch
    # forward+backward recordings pin multiple GiB of activation pools. The
    # resulting importance-ratio drift against stored behavior likelihoods is
    # gated end to end by `update_replay_parity`.
    compile_update: bool = True


@dataclass(frozen=True)
class AdvantageBatch:
    advantages: np.ndarray
    value_targets: np.ndarray


def generalized_advantage_and_targets(
    rewards: Tensor,
    values: Tensor,
    valid: Tensor,
    actor_gae_lambda: float = DEFAULT_ACTOR_GAE_LAMBDA,
    gamma: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Compute actor lambda-GAE and decoupled lambda-one critic targets."""
    if rewards.shape != values.shape or valid.shape != values.shape:
        raise ValueError("rewards, values, and valid mask must have the same shape")
    if values.ndim != 2:
        raise ValueError("values must be [trajectories, time]")
    if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be finite and in (0, 1]")
    if not math.isfinite(actor_gae_lambda) or not 0.0 <= actor_gae_lambda <= 1.0:
        raise ValueError("actor GAE lambda must be finite and in [0, 1]")
    if values.size(1) == 0:
        return torch.zeros_like(values), torch.zeros_like(values)

    valid_mask = valid.bool()
    if bool((~torch.isfinite(rewards) & valid_mask).any()):
        raise ValueError("valid rewards must be finite")
    if bool((~torch.isfinite(values) & valid_mask).any()):
        raise ValueError("valid values must be finite")
    # Invalid padding is semantically absent. Select it away before arithmetic
    # because IEEE NaN multiplied by a zero mask remains NaN and could otherwise
    # contaminate the preceding valid suffix.
    rewards = torch.where(valid_mask, rewards, torch.zeros_like(rewards))
    values = torch.where(valid_mask, values, torch.zeros_like(values))
    valid = valid_mask.to(values.dtype)
    zero_column = torch.zeros_like(values[:, :1])
    next_values = torch.cat((values[:, 1:], zero_column), dim=1)
    next_valids = torch.cat((valid[:, 1:], torch.zeros_like(valid[:, :1])), dim=1)
    deltas = rewards + gamma * next_values * next_valids - values
    running_advantage = torch.zeros(values.size(0), dtype=values.dtype, device=values.device)
    running_return = torch.zeros_like(running_advantage)
    advantage_columns: list[Tensor] = []
    target_columns: list[Tensor] = []
    for step in range(values.size(1) - 1, -1, -1):
        next_valid = next_valids[:, step]
        running_advantage = (
            deltas[:, step] + gamma * actor_gae_lambda * running_advantage * next_valid
        ) * valid[:, step]
        # This is the undiscounted Monte Carlo suffix return when gamma=1.
        # Computing it directly from rewards makes the critic target exactly
        # independent of its own predictions, including in floating point.
        running_return = (rewards[:, step] + gamma * running_return * next_valid) * valid[:, step]
        advantage_columns.append(running_advantage)
        target_columns.append(running_return)
    advantages = torch.stack(advantage_columns[::-1], dim=1).to(values.dtype)
    value_targets = torch.stack(target_columns[::-1], dim=1).to(values.dtype)
    return advantages, value_targets


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
    if config.gamma != 1.0:
        raise ValueError("Kaggriculture bank-delta rewards require undiscounted gamma=1")
    if not math.isfinite(config.actor_gae_lambda) or not 0.0 <= config.actor_gae_lambda <= 1.0:
        raise ValueError("actor GAE lambda must be finite and in [0, 1]")


def _replayed_value_chunk(
    critic: DistributionalCritic,
    board: Tensor,
    critic_features: Tensor,
    autocast_enabled: bool,
) -> Tensor:
    """One critic value forward over a chunk of stored state features.

    Runs under the same autocast state as the critic's training minibatches.
    The values feed only GAE advantages, whose tolerance is far looser than
    the ~1e-3 value shift BF16 introduces, and `critic.value` reduces the
    distributional head in fp32 either way.
    """
    with torch.autocast(
        device_type=board.device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        critic_logits = critic(board.float(), critic_features.float())
    return critic.value(critic_logits)


@torch.inference_mode()
def replay_behavior_values(
    critic: DistributionalCritic,
    board: Tensor,
    critic_features: Tensor,
    *,
    # Transient fp32 activations scale with the chunk. At production model
    # size 16384 rows would add several GiB right when the staged rollout
    # already occupies the device; 4096 keeps the pass large enough to stay
    # bandwidth-bound without that spike.
    chunk_size: int = 4096,
    compile_model: bool = False,
    autocast_enabled: bool = False,
) -> Tensor:
    """Replay behavior-time value predictions from stored state features.

    The critic is untouched between rollout collection and its first
    optimizer step of the update, so replaying the stored features through
    the critic reproduces the collection-time predictions without paying one
    small synchronous critic forward per environment step. Must run before
    the update mutates the critic. This full-batch pass is half the update's
    wall clock when run eagerly, so on CUDA it routes through the same
    Inductor compilation and autocast state as the rest of the update path.
    """
    if board.ndim < 1 or board.shape[0] != critic_features.shape[0]:
        raise ValueError("board and critic feature rows must align")
    if chunk_size < 1:
        raise ValueError("chunk size must be positive")
    forward = (
        _cached_update_callable(critic, "_kaggriculture_value_replay", _replayed_value_chunk)
        if compile_model and board.device.type == "cuda"
        else _replayed_value_chunk
    )
    was_training = critic.training
    critic.eval()
    try:
        values = [
            forward(
                critic,
                board[start : start + chunk_size],
                critic_features[start : start + chunk_size],
                autocast_enabled,
            )
            for start in range(0, board.shape[0], chunk_size)
        ]
    finally:
        critic.train(was_training)
    return torch.cat(values).float()


def prepare_advantages(
    rollout: RolloutBatch, values: np.ndarray, config: VapoConfig
) -> AdvantageBatch:
    _validate_config(config)
    if values.shape != rollout.rewards.shape:
        raise ValueError("behavior values must match the rollout reward shape")
    rewards = torch.from_numpy(rollout.rewards).float()
    values = torch.from_numpy(values).float()
    valid = torch.from_numpy(rollout.valid).float()
    advantages, targets = generalized_advantage_and_targets(
        rewards,
        values,
        valid,
        actor_gae_lambda=config.actor_gae_lambda,
        gamma=config.gamma,
    )
    selected = advantages[valid.bool()]
    if selected.numel() == 0:
        raise ValueError("rollout contains no valid states")
    normalized = (advantages - selected.mean()) / selected.std(unbiased=False).clamp_min(1e-6)
    normalized *= valid
    return AdvantageBatch(
        advantages=normalized.numpy(),
        value_targets=targets.numpy(),
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
    flat = torch.from_numpy(array.reshape((-1, *array.shape[2:])))
    # Pinned rollout arenas upload asynchronously; stream ordering keeps the
    # copies safe because every consumer runs on the same stream.
    return flat.to(device=device, non_blocking=flat.is_pinned())


def _batch_tensor(staged: Tensor, indices: Tensor, dtype: torch.dtype | None = None) -> Tensor:
    selected = staged.index_select(0, indices)
    return selected if dtype is None or selected.dtype == dtype else selected.to(dtype=dtype)


def _balanced_minibatch_slices(sample_count: int, maximum_size: int) -> tuple[slice, ...]:
    """Partition an epoch into near-equal, nonempty minibatches.

    A short final tail would otherwise receive a full optimizer step despite
    its mean loss containing fewer samples. Balancing keeps every sample's
    per-epoch influence approximately equal without dropping any states.
    """
    if sample_count < 1 or maximum_size < 1:
        raise ValueError("sample count and maximum minibatch size must be positive")
    batch_count = math.ceil(sample_count / maximum_size)
    base_size, larger_batches = divmod(sample_count, batch_count)
    slices: list[slice] = []
    start = 0
    for batch in range(batch_count):
        size = base_size + int(batch < larger_batches)
        slices.append(slice(start, start + size))
        start += size
    return tuple(slices)


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


def _replayed_component_logprobs(
    actor: FarmActor,
    board: Tensor,
    global_features: Tensor,
    units: Tensor,
    positions: Tensor,
    unit_actions: Tensor,
    market_kinds: Tensor,
    market_quantities: Tensor,
    unit_masks: Tensor,
    kind_masks: Tensor,
    quantity_masks: Tensor,
    autocast_enabled: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Replay stored actions through the update-path policy forward.

    This defines the current-likelihood side of the PPO importance ratio in
    every actor minibatch. The behavior side is produced at unchanged weights
    by `_replayed_selected_logprobs`, which runs the identical forward,
    masking, fp32 log_softmax, and gather math minus the entropy branch, so
    the ratio starts at one up to numerics. The residual — separate Inductor
    graphs when compiled, and minibatch composition that differs by the
    update loop's shuffle (row counts differ by at most one) — is observed
    directly by the `first_minibatch_approx_kl` metric and gated at
    `MAX_FIRST_MINIBATCH_KL`. Autocast keeps log_softmax in fp32 by policy,
    so the returned log-likelihoods are full precision either way.
    """
    with torch.autocast(
        device_type=board.device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        actor_output = actor(board, global_features, units, positions)
        return component_logprobs(
            actor_output,
            actor.quantity_logits(actor_output.market_quantity_context, market_kinds),
            unit_actions,
            market_kinds,
            market_quantities,
            unit_masks,
            kind_masks,
            quantity_masks,
            validate_masks=False,
        )


def _replayed_selected_logprobs(
    actor: FarmActor,
    board: Tensor,
    global_features: Tensor,
    units: Tensor,
    positions: Tensor,
    unit_actions: Tensor,
    market_kinds: Tensor,
    market_quantities: Tensor,
    unit_masks: Tensor,
    kind_masks: Tensor,
    quantity_masks: Tensor,
    autocast_enabled: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Entropy-free `_replayed_component_logprobs` for full-batch replays.

    The behavior replay and the parity audit sweep every valid state but use
    only the gathered log-likelihoods, so this variant skips the per-head
    entropy reductions the minibatch objective needs for its metrics.
    """
    with torch.autocast(
        device_type=board.device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        actor_output = actor(board, global_features, units, positions)
        return component_selected_logprobs(
            actor_output,
            actor.quantity_logits(actor_output.market_quantity_context, market_kinds),
            unit_actions,
            market_kinds,
            market_quantities,
            unit_masks,
            kind_masks,
            quantity_masks,
            validate_masks=False,
        )


@torch.no_grad()
def replay_behavior_logprobs(
    actor: FarmActor,
    staged: dict[str, Tensor],
    valid_indices: np.ndarray,
    *,
    minibatch_size: int,
    autocast_enabled: bool,
    compile_model: bool = False,
) -> dict[str, Tensor]:
    """Recompute behavior log-likelihoods through the update-path forward.

    The rollout path samples actions from logits produced by a differently
    compiled (and differently batched) forward, so its recorded likelihoods
    differ from the update path's by kernel-selection noise. Recomputing them
    here at unchanged weights, with the update path's forward and logprob
    math at the update's precision, starts the importance ratio at one up to
    numerics — which is what makes a reduced-precision update forward legal.
    (The match is not bit-exact: the update loop shuffles its balanced
    minibatches so a row's batch size can differ by one, and the compiled
    replay and minibatch graphs are separate Inductor artifacts. That
    residual is measured by `first_minibatch_approx_kl` and gated at
    `MAX_FIRST_MINIBATCH_KL`.) The rollout-vs-update divergence becomes an
    off-policy sampling bias instead of a ratio error; `update_replay_parity`
    measures exactly that divergence.

    Must run before the first actor optimizer step. `torch.no_grad` rather
    than inference mode: the outputs are later gathered inside the autograd
    minibatch graph, which inference tensors do not permit.
    """
    if minibatch_size < 1:
        raise ValueError("minibatch size must be positive")
    if valid_indices.size == 0:
        raise ValueError("rollout contains no valid states")
    device = staged["board"].device
    replay = (
        _cached_update_callable(actor, "_kaggriculture_logprob_replay", _replayed_selected_logprobs)
        if compile_model and device.type == "cuda"
        else _replayed_selected_logprobs
    )
    rows = staged["board"].shape[0]
    replayed = {
        "old_unit_logprobs": torch.zeros(
            (rows, staged["unit_actions"].shape[1]), dtype=torch.float32, device=device
        ),
        "old_market_kind_logprobs": torch.zeros(
            (rows, staged["market_kinds"].shape[1]), dtype=torch.float32, device=device
        ),
        "old_market_quantity_logprobs": torch.zeros(
            (rows, staged["market_quantities"].shape[1]), dtype=torch.float32, device=device
        ),
    }
    ordered = torch.from_numpy(valid_indices).to(device=device)
    for batch_slice in _balanced_minibatch_slices(valid_indices.size, minibatch_size):
        indices = ordered[batch_slice]
        unit_logprobs, kind_logprobs, quantity_logprobs = replay(
            actor,
            _batch_tensor(staged["board"], indices, torch.float32),
            _batch_tensor(staged["global_features"], indices, torch.float32),
            _batch_tensor(staged["units"], indices, torch.float32),
            _batch_tensor(staged["unit_positions"], indices, torch.long),
            _batch_tensor(staged["unit_actions"], indices, torch.long),
            _batch_tensor(staged["market_kinds"], indices, torch.long),
            _batch_tensor(staged["market_quantities"], indices, torch.long),
            _batch_tensor(staged["unit_masks"], indices, torch.bool),
            _batch_tensor(staged["market_kind_masks"], indices, torch.bool),
            _batch_tensor(staged["market_quantity_masks"], indices, torch.bool),
            autocast_enabled,
        )
        for name, values in (
            ("old_unit_logprobs", unit_logprobs),
            ("old_market_kind_logprobs", kind_logprobs),
            ("old_market_quantity_logprobs", quantity_logprobs),
        ):
            replayed[name].index_copy_(0, indices, values.float())
    return replayed


def _cached_update_callable(module: torch.nn.Module, attribute: str, function):
    """Compile an update-path computation with Inductor, cached per module.

    Rollout collection deliberately uses the fusion-free cudagraphs backend,
    but graph-capturing the update's forward+backward would permanently pin
    every minibatch's activations in private pools. Inductor keeps memory
    eager-like and instead removes launch overhead by fusing the elementwise
    logprob/surrogate math; the numeric drift fusion introduces is bounded by
    the `update_replay_parity` gate. The compiled wrapper is attached outside
    the module hierarchy so checkpoints stay clean.
    """
    compiled = getattr(module, attribute, None)
    if compiled is None:
        compiled = torch.compile(function, fullgraph=True, dynamic=False)
        object.__setattr__(module, attribute, compiled)
    return compiled


def _actor_minibatch_terms(
    actor: FarmActor,
    board: Tensor,
    global_features: Tensor,
    units: Tensor,
    positions: Tensor,
    unit_actions: Tensor,
    market_kinds: Tensor,
    market_quantities: Tensor,
    unit_masks: Tensor,
    kind_masks: Tensor,
    quantity_masks: Tensor,
    unit_active: Tensor,
    kind_active: Tensor,
    quantity_active: Tensor,
    old_unit: Tensor,
    old_kind: Tensor,
    old_quantity: Tensor,
    advantages: Tensor,
    clip_low: float,
    clip_high: float,
    autocast_enabled: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """One actor minibatch: policy replay plus clipped surrogate reductions.

    Returns device-side (policy objective sum, entropy sum, k3 KL sum, clipped
    count). Normalization by the per-minibatch component count stays outside
    so the varying host integer never enters the captured graph.
    """
    (
        new_unit,
        new_kind,
        new_quantity,
        unit_entropy,
        kind_entropy,
        quantity_entropy,
    ) = _replayed_component_logprobs(
        actor,
        board,
        global_features,
        units,
        positions,
        unit_actions,
        market_kinds,
        market_quantities,
        unit_masks,
        kind_masks,
        quantity_masks,
        autocast_enabled,
    )
    policy_sum = torch.zeros((), device=board.device)
    entropy_sum = torch.zeros((), device=board.device)
    kl_sum = torch.zeros((), device=board.device)
    clipped_sum = torch.zeros((), device=board.device)
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
            clip_low,
            clip_high,
        )
        policy_sum = policy_sum + component_objective
        entropy_sum = entropy_sum + (entropy.detach() * active).sum()
        kl_sum = kl_sum + component_kl
        clipped_sum = clipped_sum + component_clipped
    return policy_sum, entropy_sum, kl_sum, clipped_sum


def _critic_minibatch_loss(
    critic: DistributionalCritic,
    board: Tensor,
    critic_features: Tensor,
    value_targets: Tensor,
    autocast_enabled: bool,
) -> Tensor:
    with torch.autocast(
        device_type=board.device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        critic_logits = critic(board, critic_features)
    return distributional_value_loss(
        critic_logits,
        value_targets,
        critic.support,
        sigma_ratio=critic.config.value_sigma_ratio,
        validate=False,
    ).mean()


@torch.inference_mode()
def update_replay_parity(
    actor: FarmActor,
    rollout: RolloutBatch,
    *,
    minibatch_size: int,
    compile_model: bool = False,
    autocast_enabled: bool = False,
) -> dict[str, float | int]:
    """Measure rollout-sampling versus update-replay likelihood divergence.

    Runs the same staging, minibatch slicing, and policy forward as
    `update_vapo`'s behavior replay and compares its log-likelihoods with the
    likelihoods the rollout sampler actually drew actions from. Since
    `replay_behavior_logprobs` pins the update's importance ratio to one at
    unchanged weights by construction, this difference no longer enters the
    objective as ratio error; it instead bounds the off-policy sampling bias
    between the distribution actions were drawn from and the distribution the
    gradient assumes. Pass the production `use_bfloat16` flag so the audited
    path is the deployed one.
    """
    if minibatch_size < 1:
        raise ValueError("minibatch size must be positive")
    device = next(actor.parameters()).device
    flat_valid = rollout.valid.reshape(-1)
    valid_indices = np.flatnonzero(flat_valid)
    if valid_indices.size == 0:
        raise ValueError("rollout contains no valid states")
    staged = {
        name: _stage_tensor(getattr(rollout, name), device)
        for name in (
            "board",
            "global_features",
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
        )
    }
    ordered = torch.from_numpy(valid_indices).to(device=device)
    compile_enabled = compile_model and device.type == "cuda"
    replay = (
        _cached_update_callable(actor, "_kaggriculture_logprob_replay", _replayed_selected_logprobs)
        if compile_enabled
        else _replayed_selected_logprobs
    )
    maximum_logprob_error = dict.fromkeys(("unit", "kind", "quantity"), 0.0)
    maximum_ratio_error = dict.fromkeys(("unit", "kind", "quantity"), 0.0)
    active_counts = dict.fromkeys(("unit", "kind", "quantity"), 0)
    for batch_slice in _balanced_minibatch_slices(valid_indices.size, minibatch_size):
        indices = ordered[batch_slice]
        market_kinds = _batch_tensor(staged["market_kinds"], indices, torch.long)
        replayed = replay(
            actor,
            _batch_tensor(staged["board"], indices, torch.float32),
            _batch_tensor(staged["global_features"], indices, torch.float32),
            _batch_tensor(staged["units"], indices, torch.float32),
            _batch_tensor(staged["unit_positions"], indices, torch.long),
            _batch_tensor(staged["unit_actions"], indices, torch.long),
            market_kinds,
            _batch_tensor(staged["market_quantities"], indices, torch.long),
            _batch_tensor(staged["unit_masks"], indices, torch.bool),
            _batch_tensor(staged["market_kind_masks"], indices, torch.bool),
            _batch_tensor(staged["market_quantity_masks"], indices, torch.bool),
            autocast_enabled,
        )
        for name, new_logprobs, old_key, active_key in (
            ("unit", replayed[0], "old_unit_logprobs", "unit_active"),
            ("kind", replayed[1], "old_market_kind_logprobs", "market_active"),
            ("quantity", replayed[2], "old_market_quantity_logprobs", "market_quantity_active"),
        ):
            active = _batch_tensor(staged[active_key], indices, torch.bool)
            active_counts[name] += int(active.sum())
            if not bool(active.any()):
                continue
            difference = (
                new_logprobs[active].float()
                - _batch_tensor(staged[old_key], indices, torch.float32)[active]
            )
            if not bool(torch.isfinite(difference).all()):
                raise FloatingPointError(f"non-finite {name} update replay difference")
            maximum_logprob_error[name] = max(
                maximum_logprob_error[name], float(difference.abs().max())
            )
            maximum_ratio_error[name] = max(
                maximum_ratio_error[name], float((difference.exp() - 1.0).abs().max())
            )
    return {
        **{
            f"update_replay_{name}_logprob_max_abs_error": value
            for name, value in maximum_logprob_error.items()
        },
        **{
            f"update_replay_{name}_ratio_max_abs_error": value
            for name, value in maximum_ratio_error.items()
        },
        **{f"update_replay_{name}_active_count": value for name, value in active_counts.items()},
        "update_replay_max_ratio_error": max(maximum_ratio_error.values()),
    }


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


def _validate_staged_action_masks(staged: dict[str, Tensor], valid: Tensor) -> None:
    """Validate stored categorical support in one staged accelerator pass."""
    flags: list[Tensor] = []
    messages: list[str] = []
    for name, masks_key, actions_key in (
        ("unit", "unit_masks", "unit_actions"),
        ("market kind", "market_kind_masks", "market_kinds"),
        ("market quantity", "market_quantity_masks", "market_quantities"),
    ):
        masks = staged[masks_key]
        actions = staged[actions_key].long()
        if masks.shape[:-1] != actions.shape:
            raise ValueError(f"{name} action and mask shapes do not align")
        categories = masks.shape[-1]
        valid_rows = valid.view(valid.shape[0], *([1] * (actions.ndim - 1)))
        selected = torch.gather(masks, -1, actions.clamp(0, categories - 1).unsqueeze(-1)).squeeze(
            -1
        )
        flags.extend(
            (
                (((actions < 0) | (actions >= categories)) & valid_rows).any(),
                (~masks.any(dim=-1) & valid_rows).any(),
                (~selected & valid_rows).any(),
            )
        )
        messages.extend(
            (
                f"{name} action is outside its categorical support",
                f"{name} mask has no valid category",
                f"{name} action is masked out",
            )
        )
    # One aggregated host readback replaces per-field synchronizing checks.
    failures = torch.stack(flags).cpu()
    for failed, message in zip(failures.tolist(), messages, strict=True):
        if failed:
            raise ValueError(message)


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
    flat_valid = rollout.valid.reshape(-1)
    valid_indices = np.flatnonzero(flat_valid)
    flat_component_counts = (
        rollout.unit_active.reshape(flat_valid.size, -1).sum(axis=1, dtype=np.int64)
        + rollout.market_active.reshape(flat_valid.size, -1).sum(axis=1, dtype=np.int64)
        + rollout.market_quantity_active.reshape(flat_valid.size, -1).sum(axis=1, dtype=np.int64)
    )

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
    }
    # Stored categorical support is validated in one staged pass; repeated
    # NumPy sweeps over the multi-gigabyte host rollout would stall the update.
    _validate_staged_action_masks(staged, torch.from_numpy(flat_valid).to(device))
    # Behavior values for GAE are replayed here from the staged features at
    # full batch instead of one small synchronous critic forward per rollout
    # step. The critic still holds exactly the behavior weights at this point.
    autocast_enabled = config.use_bfloat16 and device.type == "cuda"
    compile_enabled = config.compile_update and device.type == "cuda"
    behavior_values = (
        replay_behavior_values(
            critic,
            staged["board"],
            staged["critic_features"],
            compile_model=config.compile_update,
            autocast_enabled=autocast_enabled,
        )
        .cpu()
        .numpy()
        .reshape(rollout.rewards.shape)
    )
    # Behavior likelihoods are recomputed through the update path itself (same
    # callable, precision, and minibatch partitioning as the loop below), not
    # taken from the rollout's sampling-path logits. The stored rollout
    # likelihoods remain the sampling ground truth that `update_replay_parity`
    # audits this replay against.
    staged.update(
        replay_behavior_logprobs(
            actor,
            staged,
            valid_indices,
            minibatch_size=config.minibatch_size,
            autocast_enabled=autocast_enabled,
            compile_model=config.compile_update,
        )
    )
    actor.train()
    critic.train()
    prepared = prepare_advantages(rollout, behavior_values, config)
    valid_value_targets = prepared.value_targets[rollout.valid]
    value_support = critic.support.detach().float().cpu().numpy()
    support_widths = np.diff(value_support)
    if (
        not np.isfinite(value_support).all()
        or not (support_widths > 0).all()
        or not np.allclose(support_widths, support_widths[:1])
    ):
        raise ValueError("critic value support must be finite, increasing, and evenly spaced")
    if not np.isfinite(valid_value_targets).all():
        raise ValueError("value targets must be finite")
    if (
        valid_value_targets.min() < value_support[0]
        or valid_value_targets.max() > value_support[-1]
    ):
        raise ValueError("value targets fall outside the critic support")
    staged["advantages"] = torch.from_numpy(prepared.advantages.reshape(-1)).to(device)
    staged["value_targets"] = torch.from_numpy(prepared.value_targets.reshape(-1)).to(device)
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
    actor_states = 0
    total_components = 0
    updates = 0
    actor_updates = 0
    completed_epochs = 0
    max_approx_kl = 0.0
    first_minibatch_kl = 0.0
    actor_terms = (
        _cached_update_callable(actor, "_kaggriculture_update_terms", _actor_minibatch_terms)
        if compile_enabled
        else _actor_minibatch_terms
    )
    critic_loss_fn = (
        _cached_update_callable(critic, "_kaggriculture_update_loss", _critic_minibatch_loss)
        if compile_enabled
        else _critic_minibatch_loss
    )
    stop_for_kl = False
    # Guard scalars leave the device through one pinned async copy per
    # minibatch. A CUDA event scopes the host wait to that tiny copy, so the
    # KL/finiteness decisions overlap the already-queued critic backward
    # instead of serializing the stream after every actor forward.
    guard_host = torch.empty(3, dtype=torch.float64, pin_memory=device.type == "cuda")
    guard_event = torch.cuda.Event() if device.type == "cuda" else None
    zero_guard = torch.zeros((), dtype=torch.float64, device=device)

    for _epoch in range(config.epochs):
        shuffled = generator.permutation(valid_indices)
        shuffled_device = torch.from_numpy(shuffled).to(device=device)
        for batch_slice in _balanced_minibatch_slices(shuffled.size, config.minibatch_size):
            host_indices = shuffled[batch_slice]
            indices = shuffled_device[batch_slice]
            board = _batch_tensor(staged["board"], indices, torch.float32)
            critic_features = _batch_tensor(staged["critic_features"], indices, torch.float32)
            value_targets = _batch_tensor(staged["value_targets"], indices, torch.float32)
            states = indices.numel()

            run_actor = not stop_for_kl
            if run_actor:
                # Component activity is immutable rollout metadata. Reducing it
                # on the host avoids a CUDA synchronization in every minibatch
                # merely to recover a denominator already known before staging.
                component_count = max(1, int(flat_component_counts[host_indices].sum()))
                global_features = _batch_tensor(staged["global_features"], indices, torch.float32)
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

                actor_optimizer.zero_grad(set_to_none=True)
                # Behavior likelihoods were replayed above at unchanged
                # weights through the update path's logprob math, so the ratio
                # starts at one up to numerics (separate compiled graphs and
                # shuffle-dependent batch composition); the first-minibatch KL
                # metric observes that residual.
                policy_sum, entropy_sum, kl_sum, clipped_sum = actor_terms(
                    actor,
                    board,
                    global_features,
                    units,
                    positions,
                    unit_actions,
                    market_kinds,
                    market_quantities,
                    unit_masks,
                    kind_masks,
                    quantity_masks,
                    unit_active,
                    kind_active,
                    quantity_active,
                    old_unit,
                    old_kind,
                    old_quantity,
                    advantages,
                    config.clip_low,
                    config.clip_high,
                    autocast_enabled,
                )
                batch_kl = kl_sum.detach().double() / component_count
                policy_loss = -policy_sum / component_count
                entropy_mean = entropy_sum / component_count
                # Gradients are computed eagerly but the actor is mutated only
                # after the deferred trust-region check below, so the guard
                # semantics stay exact: a violating minibatch is never applied.
                policy_loss.backward()
                actor_gradient_norm = torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), config.max_gradient_norm
                ).detach()

            # Target KL constrains only the actor. Keep fitting the critic for
            # every configured epoch even after policy replay is frozen.
            critic_optimizer.zero_grad(set_to_none=True)
            value_loss = critic_loss_fn(
                critic, board, critic_features, value_targets, autocast_enabled
            )
            guard_values = torch.stack(
                (
                    batch_kl if run_actor else zero_guard,
                    policy_loss.detach().double() if run_actor else zero_guard,
                    value_loss.detach().double(),
                )
            )
            guard_host.copy_(guard_values, non_blocking=True)
            if guard_event is not None:
                guard_event.record()
            value_loss.backward()
            critic_gradient_norm = torch.nn.utils.clip_grad_norm_(
                critic.parameters(), config.max_gradient_norm
            ).detach()

            # The event covers only the three-scalar copy, so this wait
            # overlaps the critic backward still executing on the stream.
            if guard_event is not None:
                guard_event.synchronize()
            batch_kl_value, policy_loss_value, value_loss_value = guard_host.tolist()
            if run_actor:
                if updates == 0:
                    # At unchanged weights this KL is pure numerics: the drift
                    # between the behavior replay above and this minibatch
                    # forward.
                    first_minibatch_kl = batch_kl_value
                max_approx_kl = max(max_approx_kl, batch_kl_value)
                # Non-finite losses abort training; the already-queued backward
                # of a poisoned minibatch is never observed past this raise.
                if not math.isfinite(policy_loss_value):
                    raise FloatingPointError("non-finite policy loss")
                # The KL belongs to the policy that produced these gradients,
                # so enforce the trust region before mutating that policy.
                if batch_kl_value > config.target_kl:
                    stop_for_kl = True
                else:
                    _optimizer_step(
                        actor_optimizer,
                        config.actor_learning_rate,
                        config.lr_warmup_steps,
                    )
                    totals["policy_loss"] += policy_loss.detach().double() * component_count
                    totals["entropy"] += entropy_mean.detach().double() * component_count
                    totals["approx_kl"] += batch_kl * component_count
                    totals["clip_fraction"] += clipped_sum.detach().double()
                    totals["actor_gradient_norm"] += actor_gradient_norm * states
                    total_components += component_count
                    actor_states += states
                    actor_updates += 1
            if not math.isfinite(value_loss_value):
                raise FloatingPointError("non-finite critic loss")
            _optimizer_step(
                critic_optimizer,
                config.critic_learning_rate,
                config.lr_warmup_steps,
            )

            totals["value_loss"] += value_loss.detach().double() * states
            totals["critic_gradient_norm"] += critic_gradient_norm * states
            total_states += states
            updates += 1
        completed_epochs += 1

    metrics: dict[str, float | int] = {
        "updates": updates,
        "actor_updates": actor_updates,
        "epochs": completed_epochs,
        "states": rollout.states,
        "policy_loss": float(totals["policy_loss"] / max(1, total_components)),
        "value_loss": float(totals["value_loss"] / max(1, total_states)),
        "entropy": float(totals["entropy"] / max(1, total_components)),
        "approx_kl": float(totals["approx_kl"] / max(1, total_components)),
        "max_approx_kl": max_approx_kl,
        "first_minibatch_approx_kl": first_minibatch_kl,
        "kl_early_stop": int(stop_for_kl),
        "clip_fraction": float(totals["clip_fraction"] / max(1, total_components)),
        "actor_gradient_norm": float(totals["actor_gradient_norm"] / max(1, actor_states)),
        "critic_gradient_norm": float(totals["critic_gradient_norm"] / max(1, total_states)),
        "advantage_mean": float(prepared.advantages[rollout.valid].mean()),
        "advantage_std": float(prepared.advantages[rollout.valid].std()),
        "value_target_mean": float(prepared.value_targets[rollout.valid].mean()),
        "value_target_std": float(prepared.value_targets[rollout.valid].std()),
        "value_target_min": float(prepared.value_targets[rollout.valid].min()),
        "value_target_max": float(prepared.value_targets[rollout.valid].max()),
        "actor_gae_lambda": config.actor_gae_lambda,
        "critic_gae_lambda": CRITIC_GAE_LAMBDA,
        "gamma": config.gamma,
        "actor_learning_rate": float(actor_optimizer.param_groups[0]["lr"]),
        "critic_learning_rate": float(critic_optimizer.param_groups[0]["lr"]),
        "explained_variance": _explained_variance(
            prepared.value_targets, behavior_values, rollout.valid
        ),
    }
    return metrics
