from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

import kaggriculture.ppo as ppo
from kaggriculture.model import DistributionalCritic, FarmActor, ModelConfig
from kaggriculture.rollout import collect_self_play


def test_padded_component_policy_loss_gradient_and_metrics_ignore_nonfinite_entries() -> None:
    generator = torch.Generator().manual_seed(301)
    logits = torch.randn(5, 6, generator=generator, requires_grad=True)
    old = logits.detach() - torch.tensor([[0.02], [-0.4], [0.7], [-0.8], [0.03]])
    advantages = torch.tensor([-7.0, 12.0, -4.0, 5.0, 0.6])
    active = torch.tensor(
        [
            [1, 1, 0, 1, 0, 0],
            [1, 0, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 0, 1],
        ],
        dtype=torch.float32,
    )
    entropy = torch.rand(5, 6, generator=generator)
    with torch.no_grad():
        logits[:4] = float("nan")
        logits.masked_fill_(~active.bool(), float("nan"))
    old[:4] = float("nan")
    old.masked_fill_(~active.bool(), float("nan"))
    advantages[:4] = float("nan")
    entropy[:4] = float("nan")
    entropy.masked_fill_(~active.bool(), float("nan"))
    positions, _counts = ppo._fixed_minibatch_positions(5, 4)

    def terms(indices: torch.Tensor, weight: torch.Tensor | None = None):
        return ppo._component_policy_sums(
            tuple(logits[indices].split(2, dim=-1)),
            tuple(old[indices].split(2, dim=-1)),
            tuple(active[indices].split(2, dim=-1)),
            tuple(entropy[indices].split(2, dim=-1)),
            advantages[indices],
            0.8,
            1.28,
            sample_weight=weight,
        )

    padded = terms(torch.from_numpy(positions[-1]), torch.tensor([1.0, 0.0, 0.0, 0.0]))
    reference = terms(torch.tensor([4]))
    for actual, expected in zip(padded, reference, strict=True):
        torch.testing.assert_close(actual, expected)
    component_count = active[4].sum()
    padded_gradient = torch.autograd.grad(-padded[0] / component_count, logits)[0]
    reference_gradient = torch.autograd.grad(-reference[0] / component_count, logits)[0]
    torch.testing.assert_close(padded_gradient, reference_gradient)
    assert torch.count_nonzero(padded_gradient[:4]) == 0


@pytest.mark.parametrize("scalar_value", [False, True])
def test_padded_value_loss_gradient_and_fit_moments_match_real_rows(scalar_value: bool) -> None:
    critic = DistributionalCritic(
        ModelConfig(
            cnn_width=8,
            cnn_blocks=1,
            model_dim=16,
            transformer_layers=3,
            attention_heads=2,
            scalar_value=scalar_value,
        )
    )
    width = 1 if scalar_value else critic.support.numel()
    logits = torch.randn(5, width, generator=torch.Generator().manual_seed(302), requires_grad=True)
    targets = torch.tensor([-1.3, 1.4, -0.8, 1.1, 0.2])
    indices = torch.tensor([4, 0, 1, 2])
    weight = torch.tensor([1.0, 0.0, 0.0, 0.0])
    padded = ppo._value_objective(critic, logits[indices], targets[indices], sample_weight=weight)
    reference = ppo._value_objective(critic, logits[4:], targets[4:])
    torch.testing.assert_close(padded, reference)
    torch.testing.assert_close(
        torch.autograd.grad(padded, logits)[0], torch.autograd.grad(reference, logits)[0]
    )
    residuals = targets.double() - critic.value(logits).detach().double()
    torch.testing.assert_close(
        ppo._value_fit_moments(targets[indices].double(), residuals[indices], weight),
        torch.stack(
            (targets[4].double(), targets[4].double().square(), residuals[4], residuals[4].square())
        ),
    )


def _update_fixture():
    torch.manual_seed(303)
    model_config = ModelConfig(
        cnn_width=8, cnn_blocks=1, model_dim=16, transformer_layers=3, attention_heads=2
    )
    actor = FarmActor(model_config)
    critic = DistributionalCritic(model_config)
    rollout = collect_self_play(actor, games=1, seed_start=304, episode_steps=5, sampling_seed=305)
    # Dense noise is not a completed-game outcome; label it as the shaped mode.
    rollout.rewards[:] = np.random.default_rng(306).normal(0, 0.1, rollout.rewards.shape)
    return actor, critic, replace(rollout, reward_mode="shaped")


def test_update_padding_preserves_parameters_and_reported_objectives() -> None:
    actor, critic, rollout = _update_fixture()
    padded_actor, padded_critic = copy.deepcopy(actor), copy.deepcopy(critic)
    count = int(rollout.valid.sum())
    config = ppo.PpoConfig(
        epochs=1,
        minibatch_size=count,
        optimizer="adamw",
        target_kl=10.0,
        use_bfloat16=False,
    )
    reference = ppo.update_ppo(
        actor,
        critic,
        *ppo.make_optimizers(actor, critic, config),
        rollout,
        config,
        generator=np.random.default_rng(307),
    )
    padded_config = replace(config, minibatch_size=count + 3)
    padded = ppo.update_ppo(
        padded_actor,
        padded_critic,
        *ppo.make_optimizers(padded_actor, padded_critic, padded_config),
        rollout,
        padded_config,
        generator=np.random.default_rng(307),
    )
    for actual_model, expected_model in ((padded_actor, actor), (padded_critic, critic)):
        for actual, expected in zip(
            actual_model.parameters(), expected_model.parameters(), strict=True
        ):
            torch.testing.assert_close(actual, expected)
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
        torch.testing.assert_close(torch.tensor(padded[key]), torch.tensor(reference[key]))


def test_update_reshuffles_each_epoch_and_counts_each_transition_once(monkeypatch) -> None:
    actor, critic, rollout = _update_fixture()
    valid = np.flatnonzero(rollout.valid.reshape(-1))
    batch_size = valid.size // 2 + 1
    config = ppo.PpoConfig(
        epochs=3,
        minibatch_size=batch_size,
        optimizer="adamw",
        target_kl=10.0,
        use_bfloat16=False,
    )
    batches = []
    original = ppo._actor_batch_args

    def record(architecture, staged, indices):
        batches.append(indices.cpu().numpy().copy())
        return original(architecture, staged, indices)

    monkeypatch.setattr(ppo, "_actor_batch_args", record)
    metrics = ppo.update_ppo(
        actor,
        critic,
        *ppo.make_optimizers(actor, critic, config),
        rollout,
        config,
        generator=np.random.default_rng(308),
    )
    _, counts = ppo._fixed_minibatch_positions(valid.size, batch_size)
    assert metrics["actor_updates"] == config.epochs * len(counts)
    assert len(batches) == config.epochs * len(counts)
    orders = []
    for epoch in range(config.epochs):
        epoch_batches = batches[epoch * len(counts) : (epoch + 1) * len(counts)]
        order = np.concatenate(
            [batch[:count] for batch, count in zip(epoch_batches, counts, strict=True)]
        )
        np.testing.assert_array_equal(np.sort(order), valid)
        orders.append(order)
    assert not np.array_equal(orders[0], orders[1])
