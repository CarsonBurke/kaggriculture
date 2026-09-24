from __future__ import annotations

import torch

from kaggriculture.entity import EntityActor, EntityConfig, EntityCritic
from kaggriculture.inference import CHECKPOINT_FORMAT_VERSION, load_actor_artifact
from kaggriculture.provenance import source_identity
from kaggriculture.registry import ENTITY_ATTENTION
from kaggriculture.structured import StructuredDecisionBelief


def test_interface_three_heads_and_all_likelihood() -> None:
    config = EntityConfig(action_interface=3)
    actor = EntityActor(config)
    belief = StructuredDecisionBelief(
        torch.zeros(2, 16, config.model_dim),
        torch.randn(2, 21, config.model_dim),
    )
    output = actor.decode_belief(belief)
    assert output.unit_logits.shape == (2, 16, 68)
    assert output.market_kind_logits.shape == (2, 21, 0)
    assert output.market_quantity_context.shape == (2, 21, config.quantity_rank)

    mask = torch.zeros(2, 21, 101, dtype=torch.bool)
    mask[..., 0] = True
    mask[..., 1:4] = True
    logits = actor.market_set_logits(output.market_quantity_context, mask)
    assert logits.shape == (2, 21, 101)
    loss = -logits[..., 3].mean()
    loss.backward()
    assert actor.market_quantity_value.weight.grad is not None
    assert actor.market_quantity_value.weight.grad[101].abs().sum() > 0


def test_interface_three_checkpoint_roundtrip_and_order_identity(tmp_path) -> None:
    config = EntityConfig(
        action_interface=3, market_set_sell_order="impact", market_set_hire_last=True
    )
    actor = EntityActor(config)
    restored = EntityActor(config)
    restored.load_state_dict(actor.state_dict(), strict=True)
    assert actor.market_kind is None
    assert actor.trunk.market_queries.num_embeddings == 21
    assert EntityCritic(config).trunk.market_queries.num_embeddings == 10
    assert actor.market_set_kind_ids.shape == (21,)
    assert actor.market_set_kind_ids[-1].item() == 1  # HIRE is last for this variant.
    path = tmp_path / "market-set-actor.pt"
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "architecture": ENTITY_ATTENTION,
            "model_config": config.to_dict(),
            "actor": actor.state_dict(),
            "source_identity": source_identity(),
        },
        path,
    )
    loaded, _ = load_actor_artifact(path)
    assert isinstance(loaded, EntityActor)
    assert loaded.config == config


def test_old_interface_heads_remain_unchanged() -> None:
    for version, rows in ((1, 100), (2, 101)):
        actor = EntityActor(EntityConfig(action_interface=version))
        assert actor.market_kind.out_features == 22
        assert actor.trunk.market_queries.num_embeddings == 10
        assert actor.market_quantity_value.num_embeddings == rows
