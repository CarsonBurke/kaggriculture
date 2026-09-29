from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from kaggriculture.actions import N_UNIT_ACTIONS, MarketKind, UnitAction


def _probe_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "probe_product_sales.py"
    spec = importlib.util.spec_from_file_location("probe_product_sales", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_action_diagnostics_excludes_inactive_and_invalid_rows() -> None:
    valid = np.zeros((2, 719), dtype=np.bool_)
    valid[:, 0] = True
    valid[0, 718] = True
    unit_active = np.zeros((2, 719, 2), dtype=np.bool_)
    market_active = np.zeros((2, 719, 2), dtype=np.bool_)
    unit_actions = np.zeros((2, 719, 2), dtype=np.int8)
    market_kinds = np.zeros((2, 719, 2), dtype=np.int8)
    unit_masks = np.zeros((2, 719, 2, N_UNIT_ACTIONS), dtype=np.bool_)

    market_active[0, 0, 0] = True
    market_active[0, 718, 0] = True
    market_active[1, 0, 1] = True
    market_kinds[0, 0, 0] = MarketKind.HIRE
    market_kinds[0, 718, 0] = MarketKind.HIRE
    market_kinds[1, 0, 1] = MarketKind.HIRE
    market_active[1, 718, 0] = True  # Invalid trajectory row.
    market_kinds[1, 718, 0] = MarketKind.HIRE

    unit_active[0, 0, :] = True
    unit_active[0, 718, 0] = True
    unit_active[1, 0, 0] = True
    unit_actions[0, 0, 0] = UnitAction.PICKUP_WHEAT_16
    unit_actions[0, 0, 1] = UnitAction.PLACE_WHEAT
    unit_actions[0, 718, 0] = UnitAction.FEED
    unit_actions[1, 0, 0] = UnitAction.PICKUP_FERTILIZER_8
    # A pickup is legal when the shed holds at least its quantity, so a legal
    # cap bin implies every smaller bin of its family (core.rs `pickup_spec`).
    wheat = slice(UnitAction.PICKUP_WHEAT_1, UnitAction.PICKUP_WHEAT_16 + 1)
    fertilizer = slice(UnitAction.PICKUP_FERTILIZER_1, UnitAction.PICKUP_FERTILIZER_8 + 1)
    unit_masks[0, 0, :, wheat] = True
    unit_masks[0, 718, 0, fertilizer] = True
    unit_masks[1, 0, 0, fertilizer] = True
    unit_masks[1, 718, 0, wheat] = True

    rollout = SimpleNamespace(
        valid=valid,
        unit_active=unit_active,
        market_active=market_active,
        unit_actions=unit_actions,
        market_kinds=market_kinds,
        unit_masks=unit_masks,
    )
    report = _probe_module().action_diagnostics(rollout)

    assert report["hire"] == {
        "orders": 3,
        "by_step": {"0": 2, "718": 1},
        "terminal_step_718": 1,
    }
    assert report["pickup_caps"]["wheat_16"] == {
        "selected": 1,
        "legal_opportunities": 2,
    }
    assert report["pickup_caps"]["fertilizer_8"] == {
        "selected": 1,
        "legal_opportunities": 2,
    }
    assert report["place_wheat"] == 1
    assert report["feed"] == 1
    # Only valid, active rows count; each selected pickup took its cap.
    assert report["pickup_amounts"]["wheat"] == {
        "selected": 1,
        "legal_opportunities": 2,
        "nonmaximum_selected": 0,
        "units": 16,
        "amounts": {"16": 1},
    }
    assert report["pickup_amounts"]["fertilizer"]["amounts"] == {"8": 1}
