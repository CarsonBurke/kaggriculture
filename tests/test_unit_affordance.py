import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from kaggle_environments import make

from kaggriculture.actions import N_UNIT_ACTIONS, UnitAction
from kaggriculture.constants import MAX_MARKET_ORDERS, MAX_UNITS
from kaggriculture.lejepa import JepaObjective
from kaggriculture.lejepa_model import LejepaActor, LejepaConfig, _UnitAffordanceScorer
from kaggriculture.ppo import _replayed_selected_logprobs
from kaggriculture.provenance import source_identity
from kaggriculture.rust_env import load_native
from kaggriculture.structured import StructuredInputs, stack_structured
from kaggriculture.tokens import (
    ANIMAL_TOKEN_FIELDS,
    CROP_TOKEN_FIELDS,
    FARM_TOKEN_FIELDS,
    N_TILE_CONTINUOUS,
    N_UNIT_CONTINUOUS,
    PRODUCT_TOKEN_FIELDS,
    TILE_COUNT,
    TOWN_TOKEN_FIELDS,
    encode_structured_observation,
)


def inputs() -> StructuredInputs:
    batch = 1
    gather = torch.zeros(batch, MAX_UNITS, 5, dtype=torch.long)
    gather[:, 0] = torch.tensor([44, 34, 54, 45, 43])
    active = torch.zeros(batch, MAX_UNITS, dtype=torch.bool)
    active[:, 0] = True
    valid = torch.zeros(batch, MAX_UNITS, 5, dtype=torch.bool)
    valid[:, 0] = True
    return StructuredInputs(
        tile_categorical=torch.zeros(batch, 2 * TILE_COUNT, 6, dtype=torch.long),
        tile_continuous=torch.zeros(batch, 2 * TILE_COUNT, N_TILE_CONTINUOUS),
        unit_categorical=torch.zeros(batch, MAX_UNITS, 4, dtype=torch.long),
        unit_continuous=torch.zeros(batch, MAX_UNITS, N_UNIT_CONTINUOUS),
        unit_active=active,
        unit_tile_gather=gather,
        unit_tile_gather_valid=valid,
        products=torch.zeros(batch, 9, len(PRODUCT_TOKEN_FIELDS)),
        animals=torch.zeros(batch, 3, len(ANIMAL_TOKEN_FIELDS)),
        crops=torch.zeros(batch, 5, len(CROP_TOKEN_FIELDS)),
        farms=torch.zeros(batch, 2, len(FARM_TOKEN_FIELDS)),
        town=torch.zeros(batch, len(TOWN_TOKEN_FIELDS)),
    )


def tiny_config(**kwargs) -> LejepaConfig:
    return LejepaConfig(
        action_interface=2,
        model_dim=32,
        attention_heads=2,
        attention_kv_heads=1,
        ffn_multiplier=2,
        farm_blocks=1,
        core_layers=2,
        jepa_hidden_dim=48,
        jepa_slices=16,
        jepa_tile_samples=8,
        jepa_sigreg_rows=64,
        **kwargs,
    )


def test_zero_residual_reproduces_all_head_logits_exactly() -> None:
    torch.manual_seed(37)
    base = LejepaActor(tiny_config()).eval()
    arm = LejepaActor(replace(base.config, unit_affordance_scorer=True)).eval()
    result = arm.load_state_dict(base.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert result.missing_keys
    assert all(key.startswith("unit_affordance.") for key in result.missing_keys)
    observation = inputs()
    with torch.no_grad():
        original = base(observation)
        candidate = arm(observation)
    for left, right in zip(original, candidate, strict=True):
        assert torch.equal(left, right)
    assert candidate.unit_logits.shape == (1, MAX_UNITS, N_UNIT_ACTIONS)


def test_real_observation_matches_on_rollout_and_replay_paths() -> None:
    environment = make("kaggriculture", configuration={"episodeSteps": 2, "seed": 3})
    observation = environment.state[0].observation
    rows, extras = stack_structured([encode_structured_observation(observation)])
    assert extras is None
    base = LejepaActor(tiny_config()).eval()
    arm = LejepaActor(replace(base.config, unit_affordance_scorer=True)).eval()
    arm.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        for path in ("forward_with_belief", "forward_with_auxiliary_belief"):
            original = getattr(base, path)(rows)[0]
            candidate = getattr(arm, path)(rows)[0]
            for left, right in zip(original, candidate, strict=True):
                assert torch.equal(left, right)


def test_target_order_and_resource_maps() -> None:
    scorer = _UnitAffordanceScorer(4, rank=1)
    assert scorer.item[UnitAction.PICKUP_FERTILIZER_3].item() == 8
    assert scorer.amount[UnitAction.PICKUP_FERTILIZER_3].item() == 3 / 16
    assert scorer.item[UnitAction.PLACE_EGG].item() == 5
    assert scorer.crop[UnitAction.PLANT_TOMATO].item() == 2
    assert scorer.crop.max().item() == 4


def test_only_local_verbs_use_current_tile_after_training() -> None:
    scorer = _UnitAffordanceScorer(1, rank=1).eval()
    observation = inputs()
    observation.tile_continuous[0, 44, 0] = 1.0
    with torch.no_grad():
        scorer.query.weight.fill_(1)
        scorer.tile_kind.weight.zero_()
        scorer.tile_occupant.weight.zero_()
        scorer.tile_continuous.weight.zero_()
        scorer.tile_continuous.weight[0, 0] = 1.0
        scorer.action.weight.zero_()
        scorer.resource.weight.zero_()
        scores = scorer(torch.ones(1, MAX_UNITS, 1), observation)
    assert scores[0, 0, UnitAction.WATER] > 0
    assert torch.count_nonzero(scores[0, 0, : UnitAction.DROP]) == 0
    assert torch.count_nonzero(scores[0, 1:]) == 0


def test_zero_initialized_query_receives_policy_gradient() -> None:
    scorer = _UnitAffordanceScorer(4, rank=2)
    unit_states = torch.randn(1, MAX_UNITS, 4)
    scorer(unit_states, inputs())[0, 0, UnitAction.WATER].backward()
    assert scorer.query.weight.grad is not None
    assert torch.count_nonzero(scorer.query.weight.grad) > 0


def test_promoted_ppo_objective_warm_starts_same_affordance_config(tmp_path: Path) -> None:
    config = tiny_config(unit_affordance_scorer=True)
    pretrained = LejepaActor(config)
    objective = JepaObjective(config)
    artifact = tmp_path / "affordance-ppo.pt"
    torch.save(
        {
            "format_version": 17,
            "architecture": "lejepa",
            "model_config": config.to_dict(),
            "actor": pretrained.state_dict(),
            "structured_dynamics": objective.state_dict(),
            "iteration": 5,
            "source_identity": source_identity(),
            "run_provenance": None,
            "seed_usage": [{"domain": "online_rl", "start": 20_500_000, "count": 16}],
        },
        artifact,
    )
    script = Path(__file__).parents[1] / "scripts" / "train_ppo.py"
    spec = importlib.util.spec_from_file_location("affordance_train_ppo", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    warm_actor = LejepaActor(config)
    warm_objective = JepaObjective(config)
    module._load_initial_actor(
        artifact, warm_actor, "lejepa", config, torch.device("cpu"), warm_objective
    )
    for key, value in pretrained.state_dict().items():
        assert torch.equal(warm_actor.state_dict()[key], value)
    for key, value in objective.state_dict().items():
        assert torch.equal(warm_objective.state_dict()[key], value)


def test_nonzero_scorer_matches_both_seat_encodings_and_native_replay() -> None:
    seed = 84631
    environment = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    environment.reset(2)
    native = load_native(build=False).BatchEnv(np.asarray([seed], dtype=np.uint64))
    encoded = native.structured()
    official = [
        encode_structured_observation(environment.state[seat].observation)
        for seat in range(2)
    ]
    for name in StructuredInputs._fields:
        np.testing.assert_array_equal(
            np.stack([getattr(row, name) for row in official]),
            np.asarray(encoded[name]),
            err_msg=f"seat-relative Python/native feature mismatch: {name}",
        )
    python_inputs, extras = stack_structured(official)
    assert extras is None
    integer_fields = {"tile_categorical", "unit_categorical", "unit_tile_gather"}
    native_inputs = StructuredInputs(
        **{
            name: torch.as_tensor(
                np.asarray(encoded[name]),
                dtype=(
                    torch.int64 if name in integer_fields else getattr(python_inputs, name).dtype
                ),
            )
            for name in StructuredInputs._fields
        }
    )
    actor = LejepaActor(tiny_config(unit_affordance_scorer=True)).eval()
    assert actor.unit_affordance is not None
    baseline = LejepaActor(replace(actor.config, unit_affordance_scorer=False)).eval()
    baseline.load_state_dict(
        {
            key: value
            for key, value in actor.state_dict().items()
            if not key.startswith("unit_affordance.")
        }
    )
    with torch.no_grad():
        actor.unit_affordance.query.weight.fill_(0.1)
        python_output = actor(python_inputs)
        native_output = actor(native_inputs)
        for left, right in zip(python_output, native_output, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        original = baseline(native_inputs)
        assert torch.equal(
            original.unit_logits[..., : UnitAction.DROP],
            native_output.unit_logits[..., : UnitAction.DROP],
        )
        assert torch.any(
            original.unit_logits[..., UnitAction.DROP :]
            != native_output.unit_logits[..., UnitAction.DROP :]
        )

        gate = actor.market_quantity_kind_gate.weight[None].float().numpy()
        values = actor.market_quantity_value.weight[None].float().numpy()
        bias = actor.market_quantity_bias[None].float().numpy()
        sampled = native.sample_and_step(
            np.ascontiguousarray(native_output.unit_logits.numpy(), dtype=np.float32),
            np.ascontiguousarray(native_output.market_kind_logits.numpy(), dtype=np.float32),
            np.ascontiguousarray(native_output.market_quantity_context.numpy(), dtype=np.float32),
            np.ascontiguousarray(gate),
            np.ascontiguousarray(values),
            np.ascontiguousarray(bias),
            np.zeros(2, dtype=np.uint16),
            np.full((2, MAX_UNITS), 0.37, dtype=np.float32),
            np.full((2, MAX_MARKET_ORDERS), 0.37, dtype=np.float32),
            np.full((2, MAX_MARKET_ORDERS), 0.37, dtype=np.float32),
            np.zeros(2, dtype=np.bool_),
            np.ones(2, dtype=np.float32),
            np.zeros(2, dtype=np.uint8),
        )
        action_names = ("unit_actions", "market_kinds", "market_quantities")
        mask_names = ("unit_masks", "market_kind_masks", "market_quantity_masks")
        actions = [
            torch.as_tensor(np.asarray(sampled[name]).astype(np.int64))
            for name in action_names
        ]
        masks = [torch.as_tensor(np.asarray(sampled[name])) for name in mask_names]
        replay = _replayed_selected_logprobs(
            actor, *actions, *masks, False, native_inputs
        )
        auxiliary_output = actor.forward_with_auxiliary_belief(native_inputs)[0]
        for rollout_tensor, auxiliary_tensor in zip(
            native_output, auxiliary_output, strict=True
        ):
            torch.testing.assert_close(rollout_tensor, auxiliary_tensor, rtol=0, atol=0)
        active_names = ("unit_active", "market_active", "market_quantity_active")
        old_names = ("unit_logprobs", "market_kind_logprobs", "market_quantity_logprobs")
        for current, active_name, old_name in zip(replay, active_names, old_names, strict=True):
            active = torch.as_tensor(np.asarray(sampled[active_name]))
            old = torch.as_tensor(np.asarray(sampled[old_name]))
            torch.testing.assert_close(current[active], old[active], rtol=0, atol=2e-4)
