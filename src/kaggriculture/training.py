"""Checkpointing and diagnostics for long-running VAPO jobs."""

from __future__ import annotations

import json
import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kaggriculture.actions import MarketKind, UnitAction
from kaggriculture.constants import QUANTITY_BINS
from kaggriculture.model import DistributionalCritic, FarmActor, ModelConfig
from kaggriculture.rollout import RolloutBatch
from kaggriculture.vapo import VapoConfig


def rollout_diagnostics(rollout: RolloutBatch) -> dict[str, float | int]:
    margins = rollout.final_money - rollout.opponent_money
    outcomes = (margins > 0).astype(np.float32) - (margins < 0).astype(np.float32)
    active_units = rollout.unit_active
    active_markets = rollout.market_active
    active_quantities = rollout.market_quantity_active
    unit_actions = rollout.unit_actions[active_units]
    market_kinds = rollout.market_kinds[active_markets]
    market_quantities = rollout.market_quantities[active_quantities]
    return {
        "rollout_trajectories": rollout.trajectories,
        "rollout_states": rollout.states,
        "rollout_horizon": rollout.horizon,
        "rollout_seconds": rollout.elapsed_seconds,
        "rollout_states_per_second": rollout.states / max(rollout.elapsed_seconds, 1e-9),
        "rollout_entropy": rollout.mean_entropy,
        "money_mean": float(rollout.final_money.mean()),
        "money_median": float(np.median(rollout.final_money)),
        "money_min": float(rollout.final_money.min()),
        "money_max": float(rollout.final_money.max()),
        "margin_abs_mean": float(np.abs(margins).mean()),
        "score_rate": float(((outcomes + 1.0) / 2.0).mean()),
        "seat_zero_score_rate": float(
            ((outcomes[rollout.seats == 0] + 1.0) / 2.0).mean()
            if (rollout.seats == 0).any()
            else 0.0
        ),
        "seat_one_score_rate": float(
            ((outcomes[rollout.seats == 1] + 1.0) / 2.0).mean()
            if (rollout.seats == 1).any()
            else 0.0
        ),
        "tie_fraction": float((outcomes == 0).mean()),
        "unit_pass_fraction": float((unit_actions == UnitAction.PASS).mean()),
        "unit_move_fraction": float(
            ((unit_actions >= UnitAction.NORTH) & (unit_actions <= UnitAction.WEST)).mean()
        ),
        "unit_shed_fraction": float(
            ((unit_actions >= UnitAction.DROP) & (unit_actions <= UnitAction.PICKUP_SHEEP)).mean()
        ),
        "unit_place_fraction": float(
            (
                (unit_actions >= UnitAction.PLACE_GOOSE) & (unit_actions <= UnitAction.PLACE_SHEEP)
            ).mean()
        ),
        "unit_plant_fraction": float(
            (
                (unit_actions >= UnitAction.PLANT_WHEAT) & (unit_actions <= UnitAction.PLANT_MELON)
            ).mean()
        ),
        "unit_water_fraction": float((unit_actions == UnitAction.WATER).mean()),
        "unit_harvest_fraction": float((unit_actions == UnitAction.HARVEST).mean()),
        "unit_fertilize_fraction": float((unit_actions == UnitAction.FERTILIZE).mean()),
        "unit_dig_fraction": float((unit_actions == UnitAction.DIG).mean()),
        "unit_build_fraction": float(
            (
                (unit_actions == UnitAction.BUILD_COOP) | (unit_actions == UnitAction.BUILD_PASTURE)
            ).mean()
        ),
        "unit_feed_fraction": float((unit_actions == UnitAction.FEED).mean()),
        "unit_collect_fertilizer_fraction": float(
            (unit_actions == UnitAction.COLLECT_FERTILIZER).mean()
        ),
        "unit_care_fraction": float((unit_actions == UnitAction.CARE).mean()),
        "market_stop_fraction": float((market_kinds == MarketKind.STOP).mean()),
        "market_hire_fraction": float((market_kinds == MarketKind.HIRE).mean()),
        "market_buy_land_fraction": float((market_kinds == MarketKind.BUY_LAND).mean()),
        "market_buy_seed_fraction": float(
            (
                (market_kinds >= MarketKind.BUY_SEED_WHEAT)
                & (market_kinds <= MarketKind.BUY_SEED_MELON)
            ).mean()
        ),
        "market_buy_product_fraction": float(
            (
                (market_kinds >= MarketKind.BUY_PRODUCT_WHEAT)
                & (market_kinds <= MarketKind.BUY_PRODUCT_FERTILIZER)
            ).mean()
        ),
        "market_buy_animal_fraction": float(
            (
                (market_kinds >= MarketKind.BUY_ANIMAL_GOOSE)
                & (market_kinds <= MarketKind.BUY_ANIMAL_SHEEP)
            ).mean()
        ),
        "market_sell_fraction": float(
            (
                (market_kinds >= MarketKind.SELL_WHEAT)
                & (market_kinds <= MarketKind.SELL_FERTILIZER)
            ).mean()
        ),
        "market_quantity_fraction": float(active_quantities.sum() / max(1, active_markets.sum())),
        "market_quantity_mean": float(
            np.asarray(QUANTITY_BINS)[market_quantities].mean() if market_quantities.size else 0.0
        ),
    }


def save_checkpoint(
    path: Path,
    *,
    actor: FarmActor,
    critic: DistributionalCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    vapo_config: VapoConfig,
    iteration: int,
    next_seed: int,
    metrics: dict[str, Any],
    training_rng_state: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "iteration": iteration,
        "next_seed": next_seed,
        "model_config": model_config.to_dict(),
        "vapo_config": asdict(vapo_config),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "metrics": metrics,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "training_rng": training_rng_state,
    }
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
    actor: FarmActor,
    critic: DistributionalCritic,
    actor_optimizer: torch.optim.Optimizer | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
    *,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format_version") not in {1, 2}:
        raise ValueError(f"unsupported checkpoint format: {payload.get('format_version')}")
    actor.load_state_dict(payload["actor"])
    critic.load_state_dict(payload["critic"])
    if actor_optimizer is not None:
        actor_optimizer.load_state_dict(payload["actor_optimizer"])
    if critic_optimizer is not None:
        critic_optimizer.load_state_dict(payload["critic_optimizer"])
    torch.set_rng_state(payload["torch_rng"].cpu())
    if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])
    return payload


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
