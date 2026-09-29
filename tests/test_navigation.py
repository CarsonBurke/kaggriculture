"""Destination marginalization keeps the engine's action likelihood contract."""

import pytest
import torch

from kaggriculture.navigation import (
    TargetNavigation,
    assemble_unit_logits,
    destination_directions,
    marginal_movement_logits,
)


def navigation_actor_inputs():
    from kaggriculture.structured import stack_structured
    from kaggriculture.tokens import encode_structured_observation

    farms = [
        {
            "money": 3000,
            "tiles": [[None for _ in range(10)] for _ in range(10)],
            "farmer": [4, 4],
            "hands": [],
        }
        for _ in range(2)
    ]
    observation = {"player": 0, "farms": farms, "private": {"inventories": [{}]}}
    return stack_structured([encode_structured_observation(observation)])[0]


@pytest.mark.parametrize("attached", [False, True])
def test_actual_actor_navigation_respects_backbone_gradient_ownership(attached):
    from kaggriculture.lejepa_model import LejepaActor, LejepaConfig

    actor = LejepaActor(
        LejepaConfig(
            model_dim=16,
            attention_heads=2,
            attention_kv_heads=1,
            farm_blocks=1,
            core_layers=1,
            policy_shapes_backbone=attached,
        )
    )
    assert actor.navigation is not None
    assert actor.unit_head[-1].out_features == 64
    output = actor(navigation_actor_inputs())
    assert output.unit_logits.shape == (1, 16, 68)
    loss = -output.unit_logits[0, 0].log_softmax(-1)[1]
    loss.backward()
    for projection in (actor.navigation.query, actor.navigation.key):
        assert projection.weight.grad is not None
        assert projection.weight.grad.abs().sum() > 0
    backbone_grad = sum(
        p.grad.abs().sum().item() for p in actor.trunk.parameters() if p.grad is not None
    )
    assert (backbone_grad > 0) is attached


def test_actual_actor_bfloat16_logits_share_native_transfer_boundary():
    from kaggriculture.lejepa_model import LejepaActor, LejepaConfig

    actor = LejepaActor(
        LejepaConfig(
            model_dim=16,
            attention_heads=2,
            attention_kv_heads=1,
            farm_blocks=1,
            core_layers=1,
            # The affordance scorer concatenates float16 observation fields,
            # which CPU autocast cannot prioritize; it has its own replay tests.
            unit_affordance_scorer=False,
        )
    ).eval()
    inputs = navigation_actor_inputs()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        replay = actor(inputs).unit_logits
        collection = actor(inputs).unit_logits
    assert replay.dtype == torch.bfloat16
    assert torch.isfinite(replay).all()
    # Native collection copies to a BF16 buffer before transferring FP32 logits;
    # replay must see exactly those rounded logits, including pointer reductions.
    transferred = collection.to(torch.bfloat16).float()
    torch.testing.assert_close(replay.float(), transferred, atol=0, rtol=0)
    torch.testing.assert_close(
        replay.float().log_softmax(-1), transferred.log_softmax(-1), atol=0, rtol=0
    )


def test_routes_make_shortest_path_progress_for_every_origin_destination():
    directions = destination_directions()
    deltas = ((0, 0), (0, -1), (0, 1), (1, 0), (-1, 0))
    for origin in range(100):
        y, x = divmod(origin, 10)
        for target in range(100):
            ty, tx = divmod(target, 10)
            action = int(directions[origin, target])
            dx, dy = deltas[action]
            assert 0 <= x + dx < 10 and 0 <= y + dy < 10
            if origin == target:
                assert action == 0
            else:
                assert abs(tx - x - dx) + abs(ty - y - dy) == abs(tx - x) + abs(ty - y) - 1


def test_bc_and_ppo_score_sum_of_latent_destination_probabilities():
    scores = torch.linspace(-2, 2, 100, requires_grad=True)
    directions = destination_directions()[44]
    gate = torch.tensor(0.7, requires_grad=True)
    moves = marginal_movement_logits(scores, directions, gate)
    local = torch.tensor([0.3, -0.2])
    executed = torch.cat((local, moves)).log_softmax(-1)
    target_probs = scores.masked_fill(directions == 0, -torch.inf).softmax(-1)
    denominator = local.exp().sum() + gate.exp()
    for action in range(1, 5):
        expected = gate.exp() * target_probs[directions == action].sum() / denominator
        torch.testing.assert_close(executed[action + 1].exp(), expected)
    # Direction-labelled BC loss and recorded PPO log probability are the same
    # marginal event; gradients reach all compatible destinations, not a label
    # invented by a route-reconstruction heuristic.
    loss = -executed[2]
    loss.backward()
    assert torch.isfinite(scores.grad).all()
    assert (scores.grad[directions == 1] < 0).all()
    assert scores.grad[directions == 0].item() == 0


def test_edge_empty_regions_have_finite_backward_and_normalized_mass():
    scores = torch.randn(100, requires_grad=True)
    directions = destination_directions()[0]
    logits = marginal_movement_logits(scores, directions, torch.tensor(0.0))
    assert logits[0].exp() == 0 and logits[3].exp() == 0
    torch.testing.assert_close(logits.exp().sum(), torch.tensor(1.0))
    logits[1:3].sum().backward()
    assert torch.isfinite(scores.grad).all()


def test_region_size_prior_does_not_prefer_vertical_or_edge_routes():
    module = TargetNavigation(8)
    with torch.no_grad():
        module.query.weight.zero_()
    origins = torch.arange(100)
    categorical = torch.zeros(1, 100, 4, dtype=torch.long)
    categorical[0, :, 2] = origins // 10
    categorical[0, :, 3] = origins % 10
    logits = module(torch.randn(1, 100, 8), torch.randn(1, 200, 8), categorical)
    directions = destination_directions()
    legal = torch.stack([(directions == a).any(-1) for a in (1, 2, 3, 4)], -1)
    torch.testing.assert_close(logits[0].exp(), legal.float(), atol=1e-6, rtol=1e-6)


def test_local_operations_unchanged_and_opponent_tiles_not_pointer_candidates():
    module = TargetNavigation(8)
    units = torch.randn(2, 3, 8)
    tiles = torch.randn(2, 200, 8)
    positions = torch.zeros(2, 3, 4, dtype=torch.long)
    positions[..., 2:] = 4
    movement = module(units, tiles, positions)
    changed = tiles.clone()
    changed[:, 100:] += 100
    torch.testing.assert_close(module(units, changed, positions), movement, rtol=0, atol=0)
    original = torch.randn(2, 3, 68)
    local = torch.cat((original[..., :1], original[..., 5:]), dim=-1)
    output = assemble_unit_logits(local, movement)
    torch.testing.assert_close(output[..., :1], original[..., :1], rtol=0, atol=0)
    torch.testing.assert_close(output[..., 5:], original[..., 5:], rtol=0, atol=0)
    output.sum().backward()
    assert module.query.weight.grad.abs().sum() > 0
