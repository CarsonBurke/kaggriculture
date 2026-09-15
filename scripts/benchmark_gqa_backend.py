#!/usr/bin/env python3
"""Compare GQA backends through the unchanged whole-iteration benchmark.

Run only in an MLQ GPU allocation. Pass --attention-backend followed by any
scripts/benchmark_ppo_iteration.py flags; all rollout, PPO, BC-loading, numeric
checks, and report-completion logic remain owned by that benchmark.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention

from kaggriculture import entity, structured
from kaggriculture.provenance import UNCOMPILED_UPDATE_COMPILE_MODE
from kaggriculture.rollout import COMPILED_ROLLOUT_FORWARD_MODES

if __package__:
    from . import benchmark_ppo_iteration as benchmark
else:
    import benchmark_ppo_iteration as benchmark


def _flex_masked_attention(
    query, key, value, mask, *, enable_gqa, scale, folded, kernel_backend="TRITON"
):
    """Fuse key validity into scores without building dynamic block metadata."""
    batch, heads, tokens, width = query.shape
    if mask.dtype != torch.bool or mask.shape[1:3] != (1, 1):
        raise ValueError("Flex production probe requires boolean key-only masks")
    if folded and enable_gqa:
        query = query.reshape(batch, key.shape[1], (heads // key.shape[1]) * tokens, width)
        enable_gqa = False

    def score_mod(score, b, h, q, k):
        return torch.where(mask[b, 0, 0, k], score, -float("inf"))

    attended = flex_attention(
        query,
        key,
        value,
        score_mod=score_mod,
        scale=scale,
        enable_gqa=enable_gqa,
        kernel_options={"BACKEND": kernel_backend},
    )
    return attended.reshape(batch, heads, tokens, width)


def _backend_dispatch(original, backend: str, flex_kernel_backend: str = "TRITON"):
    """Compare native and folded GQA without repeating K/V or changing masks."""
    flex_masked = backend in ("flex-folded-masked-update", "flex-native-masked-update")
    selected_backend = (
        SDPBackend.FLASH_ATTENTION
        if backend.startswith("flash-") or flex_masked
        else SDPBackend.CUDNN_ATTENTION
    )
    candidate = backend != "efficient"
    folded = "-folded-" in backend or flex_masked
    all_unmasked = (
        backend.endswith("-unmasked") or backend == "flash-folded-cudnn-masked" or flex_masked
    )
    cudnn_masked = backend in ("cudnn-folded-all", "flash-folded-cudnn-masked") or flex_masked

    def attention(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        *,
        enable_gqa: bool,
        scale: float,
    ) -> Tensor:
        if flex_masked and attention_mask is not None and torch.is_grad_enabled():
            return _flex_masked_attention(
                query,
                key,
                value,
                attention_mask,
                enable_gqa=enable_gqa,
                scale=scale,
                folded=backend == "flex-folded-masked-update",
                kernel_backend=flex_kernel_backend,
            )
        if cudnn_masked and attention_mask is not None:
            return original(
                query,
                key,
                value,
                attention_mask,
                enable_gqa=enable_gqa,
                scale=scale,
                backend=SDPBackend.CUDNN_ATTENTION,
            )
        use_candidate = (
            candidate
            and attention_mask is None
            and (
                all_unmasked
                or backend == "cudnn-folded-all"
                or (query.shape[-2] == 100 and key.shape[-2] == 100)
            )
        )
        if not use_candidate or folded:
            return original(
                query,
                key,
                value,
                attention_mask,
                enable_gqa=enable_gqa,
                scale=scale,
                backend=selected_backend if use_candidate else SDPBackend.EFFICIENT_ATTENTION,
            )
        # A single allowed backend makes unsupported native calls fail loudly.
        # No head repetition, group folding, precision conversion, or fallback.
        with sdpa_kernel(selected_backend):
            return torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=0.0,
                enable_gqa=enable_gqa,
                scale=scale,
            )

    return attention


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument("--flex-kernel-backend", choices=("AUTO", "TRITON"), default="TRITON")
    parser.add_argument(
        "--attention-backend",
        choices=(
            "efficient",
            "flash-farm",
            "flash-unmasked",
            "cudnn-farm",
            "cudnn-unmasked",
            "flash-folded-farm",
            "flash-folded-unmasked",
            "cudnn-folded-farm",
            "cudnn-folded-unmasked",
            "cudnn-folded-all",
            "flash-folded-cudnn-masked",
            "flex-folded-masked-update",
            "flex-native-masked-update",
        ),
        default="efficient",
    )
    options, remaining = parser.parse_known_args()
    if "--help" in remaining or "-h" in remaining:
        parser.print_help()

    original_argv = sys.argv
    original_parse_args = benchmark.parse_args
    original_emit = benchmark.emit
    original_structured_attention = structured._fused_attention
    original_entity_attention = entity._fused_attention
    backend = options.attention_backend
    provenance = {
        "name": backend,
        "flex_kernel_backend": options.flex_kernel_backend,
        "experimental": backend != "efficient",
        "flex_scope": (
            "masked gradient-enabled updates; rollout retains folded Flash/cuDNN"
            if backend.startswith("flex-")
            else None
        ),
        "candidate_scope": (
            "none"
            if backend == "efficient"
            else "masked updates with Flex; folded Flash/cuDNN rollout"
            if backend.startswith("flex-")
            else "all attention"
            if backend in ("cudnn-folded-all", "flash-folded-cudnn-masked")
            else "all unmasked attention"
            if backend.endswith("-unmasked")
            else "unmasked query_tokens=100 and key_tokens=100"
        ),
        "gqa_layout": (
            "native Flex in masked updates; folded query groups elsewhere"
            if backend == "flex-native-masked-update"
            else "folded-query-groups"
            if backend == "efficient" or "-folded-" in backend
            else "native-unequal-heads"
        ),
        "masked_attention": (
            "Flex score_mod in updates; folded cuDNN in rollout"
            if backend.startswith("flex-")
            else "folded cuDNN GQA (no repeated K/V)"
            if backend in ("cudnn-folded-all", "flash-folded-cudnn-masked")
            else "folded efficient GQA (no repeated K/V)"
        ),
        "wrapper_sha256": benchmark.file_sha256(Path(__file__).resolve()),
        "argv": original_argv,
    }

    def parse_args():
        args = original_parse_args()
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("GQA backend comparison requires CUDA; CPU is not supported")
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise RuntimeError("GQA backend comparison requires native CUDA BF16")
        if args.no_bfloat16 or not args.rollout_bfloat16:
            raise ValueError("GQA backend comparison requires BF16 rollout and PPO updates")
        if (
            args.update_compile_mode == UNCOMPILED_UPDATE_COMPILE_MODE
            or args.rollout_forward_mode not in COMPILED_ROLLOUT_FORWARD_MODES
        ):
            raise ValueError("GQA backend comparison requires compiled rollout and PPO updates")
        if args.repeats < 2:
            raise ValueError("GQA backend comparison requires a cold and a steady iteration")
        return args

    def emit(payload: dict) -> None:
        tagged = {**payload, "attention_backend": backend}
        if payload.get("event") == "configuration":
            tagged["attention_backend_provenance"] = provenance
        original_emit(tagged)

    try:
        sys.argv = [original_argv[0], *remaining]
        benchmark.parse_args = parse_args
        benchmark.emit = emit
        structured._fused_attention = _backend_dispatch(
            original_structured_attention, backend, options.flex_kernel_backend
        )
        entity._fused_attention = _backend_dispatch(
            original_entity_attention, backend, options.flex_kernel_backend
        )
        benchmark.main()
    finally:
        sys.argv = original_argv
        benchmark.parse_args = original_parse_args
        benchmark.emit = original_emit
        structured._fused_attention = original_structured_attention
        entity._fused_attention = original_entity_attention


if __name__ == "__main__":
    main()
