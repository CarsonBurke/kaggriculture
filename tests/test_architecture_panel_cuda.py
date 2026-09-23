"""Full fixed-panel contract through the compiled native production boundary."""

import numpy as np
import pytest
import torch

from kaggriculture.architecture_panel import PANEL_POLICY, evaluate_architecture_panel
from kaggriculture.entity import EntityActor, EntityConfig, EntityCritic

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="MLQ CUDA allocation required"),
]


def test_full_panel_restores_training_state_and_reports_complete_games():
    if not torch.cuda.is_bf16_supported():
        pytest.fail("panel requires CUDA BF16")
    torch.manual_seed(20260918)
    actor = EntityActor(EntityConfig()).cuda().train()
    critic = EntityCritic(EntityConfig()).cuda().train()
    before = [parameter.detach().clone() for parameter in actor.parameters()]
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    result = evaluate_architecture_panel(actor, critic, compile_mode="reduce-overhead")
    assert actor.training and critic.training
    torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
    torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
    for initial, parameter in zip(before, actor.parameters(), strict=True):
        torch.testing.assert_close(initial, parameter, rtol=0, atol=0)
        assert parameter.grad is None
    assert len(result["panels"]) == 4
    assert result["critic_states"] == 2 * PANEL_POLICY["games_per_opponent"] * 719
    assert np.isfinite(result["critic_mse"])
    assert 0 <= result["score_rate"] <= 1
    for panel in result["panels"]:
        assert len(panel["outcomes"]) == PANEL_POLICY["games_per_opponent"]
        assert set(panel["outcomes"]) <= {-1.0, 0.0, 1.0}
        assert len(panel["seeds"]) == len(set(panel["seeds"]))
        assert panel["seats"].count(0) == panel["seats"].count(1)
