"""Two-minute eager CUDA probe of causal decoding and vmapped opponents.

This isolates policy inference; it is not a full rollout or training benchmark.
Run through mlq, with a two-minute job limit.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import torch

from kaggriculture.causal_actor import CausalActor, CausalChoice, CausalConfig
from kaggriculture.device_ledger import DeviceLedger
from kaggriculture.rollout import _StackedActorEnsemble
from kaggriculture.rust_env import load_native
from kaggriculture.structured import StructuredInputs


def timed(call, repeats: int = 2) -> list[float]:
    call()
    durations = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
    return durations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=96)
    parser.add_argument("--parallel-unit-decode", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(61903)
    games = args.games
    rows = games * 2
    env = load_native().BatchEnv(np.arange(1701, 1701 + games, dtype=np.uint64))
    for _ in range(72):
        actions = env.builtin_actions(np.full(rows, 4, dtype=np.uint8))
        env.step_factors(
            *(
                actions[name].reshape(games, 2, -1)
                for name in ("unit_actions", "market_kinds", "market_quantities")
            )
        )
    encoded = env.structured()
    index_fields = {"tile_categorical", "unit_categorical", "unit_tile_gather"}
    inputs = StructuredInputs(
        *(
            torch.as_tensor(
                encoded[name], device="cuda", dtype=torch.int64 if name in index_fields else None
            )
            for name in StructuredInputs._fields
        )
    )
    choice = CausalChoice(
        torch.as_tensor(env.policy_ledger(), device="cuda"),
        torch.full((rows, 16), -1, device="cuda", dtype=torch.int64),
        torch.full((rows, 10), -1, device="cuda", dtype=torch.int64),
        torch.full((rows, 10), -1, device="cuda", dtype=torch.int64),
        torch.rand(rows, 36, device="cuda"),
        torch.ones(rows, device="cuda"),
        torch.zeros(rows, device="cuda", dtype=torch.bool),
    )
    actor = (
        CausalActor(
            CausalConfig(
                observation_schema_version=4,
                parallel_unit_decode=args.parallel_unit_decode,
            ),
            ledger=DeviceLedger.from_native("cuda"),
        )
        .cuda()
        .eval()
    )
    result = {
        "games": games,
        "rows": rows,
        "device": torch.cuda.get_device_name(),
        "observation_schema_version": 4,
        "mode": "eager_bfloat16_inference",
        "parallel_unit_decode": args.parallel_unit_decode,
    }
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        result["single_forward_seconds"] = timed(lambda: actor(inputs, choice))
        print(json.dumps({"event": "single_forward", **result}, sort_keys=True), flush=True)
        for lanes in (2, 4):
            ensemble = _StackedActorEnsemble([actor] * lanes)
            multi_inputs = StructuredInputs(
                *(value.unsqueeze(0).expand(lanes, *value.shape) for value in inputs)
            )
            multi_choice = CausalChoice(
                *(value.unsqueeze(0).expand(lanes, *value.shape) for value in choice)
            )
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                def forward_ensemble(
                    current_ensemble=ensemble,
                    current_inputs=multi_inputs,
                    current_choice=multi_choice,
                ) -> None:
                    current_ensemble._forward(current_inputs, current_choice)

                try:
                    result[f"ensemble_{lanes}_seconds"] = timed(forward_ensemble)
                except RuntimeError as error:
                    result[f"ensemble_{lanes}_error"] = str(error)
            result[f"ensemble_{lanes}_vmap_fallback_warnings"] = sum(
                "BatchedFallback" in str(item.message) for item in recorded
            )
            print(json.dumps({"event": f"ensemble_{lanes}", **result}, sort_keys=True), flush=True)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
