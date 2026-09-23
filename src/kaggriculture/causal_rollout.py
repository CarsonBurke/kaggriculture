"""Fixed-buffer native boundary for policies that select their own action prefix."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from kaggriculture.constants import MAX_MARKET_ORDERS, MAX_UNITS
from kaggriculture.device_ledger import POLICY_LEDGER_WIDTH, validate_packed


class LedgerStaging:
    """Two pinned host states share one fixed device address for graph replay."""

    def __init__(self, rows: int, device: torch.device, minimum: int, maximum: int) -> None:
        self.host = tuple(
            torch.empty((rows, POLICY_LEDGER_WIDTH), dtype=torch.int64, pin_memory=True)
            for _ in range(2)
        )
        self.arrays = tuple(value.numpy() for value in self.host)
        self.device = torch.empty((rows, POLICY_LEDGER_WIDTH), dtype=torch.int64, device=device)
        self.minimum = minimum
        self.maximum = maximum
        self.index = 0

    def refresh(self, environment: Any, *, advance: bool = False) -> None:
        if advance:
            self.index = 1 - self.index
        environment.policy_ledger_into(self.arrays[self.index])
        validate_packed(self.arrays[self.index], self.minimum, self.maximum)
        self.device.copy_(self.host[self.index], non_blocking=True)

    @property
    def current(self) -> np.ndarray:
        return self.arrays[self.index]


class SelectedFactorTransfer:
    """Transfer selected actions and behavior statistics with a single sync.

    Integers are bounded action identifiers and booleans, so FP32 packing is
    lossless. The large exact ledger travels in the other direction as int64.
    Neither neural logits nor quantity preferences need to cross this boundary.
    """

    FIELDS = (
        "unit_actions",
        "market_kinds",
        "market_quantities",
        "factor_logprobs",
        "factor_entropies",
    )

    def __init__(self, output: Any) -> None:
        values = [getattr(output, name) for name in self.FIELDS]
        self.device = torch.empty(
            sum(value.numel() for value in values), device=values[0].device, dtype=torch.float32
        )
        self.host = torch.empty(self.device.numel(), dtype=torch.float32, pin_memory=True)
        self.factors = tuple(np.empty(tuple(value.shape), dtype=np.uint8) for value in values[:3])
        self.views: list[Tensor] = []
        self.arrays: dict[str, np.ndarray] = {}
        cursor = 0
        for name, value in zip(self.FIELDS, values, strict=True):
            end = cursor + value.numel()
            self.views.append(self.device[cursor:end].view(value.shape))
            self.arrays[name] = self.host[cursor:end].view(value.shape).numpy()
            cursor = end

    def copy(self, output: Any) -> None:
        for name, destination in zip(self.FIELDS, self.views, strict=True):
            destination.copy_(getattr(output, name))
        self.host.copy_(self.device, non_blocking=True)
        torch.cuda.current_stream(self.device.device).synchronize()
        for name, destination in zip(self.FIELDS[:3], self.factors, strict=True):
            np.copyto(destination, self.arrays[name], casting="unsafe")

    def store_statistics(self, sampled: dict[str, np.ndarray]) -> None:
        """Native stepping fills authoritative masks but has no neural likelihoods."""
        logprobs = self.arrays["factor_logprobs"]
        np.copyto(np.asarray(sampled["unit_logprobs"]), logprobs[:, :MAX_UNITS])
        np.copyto(np.asarray(sampled["market_kind_logprobs"]), logprobs[:, MAX_UNITS::2])
        np.copyto(np.asarray(sampled["market_quantity_logprobs"]), logprobs[:, MAX_UNITS + 1 :: 2])
        active = np.concatenate(
            (
                np.asarray(sampled["unit_active"]),
                np.stack(
                    (sampled["market_active"], sampled["market_quantity_active"]), axis=-1
                ).reshape(logprobs.shape[0], 2 * MAX_MARKET_ORDERS),
            ),
            axis=1,
        )
        entropies = np.where(active, self.arrays["factor_entropies"], 0).sum(axis=1)
        np.copyto(np.asarray(sampled["entropy"]), entropies / np.maximum(active.sum(axis=1), 1))
