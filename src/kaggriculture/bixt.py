"""BiXT rounds with one shared similarity matrix for both attention directions.

The reference/value construction and simultaneous updates follow BiXT sections
2.2-2.3: https://arxiv.org/html/2402.12138v2 and the authors' implementation at
https://github.com/mrkshllr/BiXT/blob/main/timm/models/bixt.py .
This adaptation retains this repository's RMSNorm, gated residuals and FFNs.
Bidirectional cross-attention uses full MHA, not the surrounding model's GQA;
latent self-attention retains the existing GQA Block. The final token-only round
omits the unused latent update, mirroring the paper's output-side specialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from kaggriculture.bixt_attention import bounded_shared_score_attention
from kaggriculture.model import Linear, RMSNorm
from kaggriculture.structured import Block, FeedForward, FusedFeedForward, GatedResidual

if TYPE_CHECKING:
    from kaggriculture.entity import EntityConfig


def shared_score_attention(
    latent_references: Tensor,
    token_references: Tensor,
    latent_values: Tensor,
    token_values: Tensor,
    token_valid: Tensor,
    *,
    output_tokens: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Return simultaneous updates from old R/V tensors shaped [B,H,L-or-N,D].

    Latents are always valid; at least one data token per example must be valid.
    Only the latent-reading normalization masks tokens. Reversing that masked
    matrix would create all-negative-infinity rows for invalid token queries.
    """
    scores = (latent_references @ token_references.transpose(-1, -2)).float()
    scores = scores * latent_references.shape[-1] ** -0.5
    latent_weights = scores.masked_fill(~token_valid[:, None, None, :], -float("inf"))
    latent_weights = latent_weights.softmax(dim=-1).to(token_values.dtype)
    token_weights = scores[:, :, :, :output_tokens].softmax(dim=-2).to(latent_values.dtype)
    latent_update = latent_weights @ token_values
    token_update = token_weights.transpose(-1, -2) @ latent_values
    token_update = torch.where(token_valid[:, None, :output_tokens, None], token_update, 0.0)
    return latent_update, token_update


def _token_read(
    latent_references: Tensor, token_references: Tensor, latent_values: Tensor
) -> Tensor:
    scores = (latent_references @ token_references.transpose(-1, -2)).float()
    scores = scores * latent_references.shape[-1] ** -0.5
    weights = scores.softmax(dim=-2).to(latent_values.dtype)
    return weights.transpose(-1, -2) @ latent_values


class BiXTRound(nn.Module):
    """Symmetric pre-norm CA/FFN followed by latent SA/FFN.

    Set ``refine_latents=False`` for the final token-output round: the token
    value projection and all latent-output parameters then do not exist. Each
    returned token is independent given the old latents, so callers may pass
    only their final decision-token slice to this last round.

    ``output_tokens`` prunes token writes/FFNs to a prefix while still reading
    every input token into the latents, useful for the penultimate round.
    """

    def __init__(
        self,
        config: EntityConfig,
        *,
        refine_latents: bool = True,
        output_tokens: int | None = None,
    ) -> None:
        super().__init__()
        if output_tokens is not None and (type(output_tokens) is not int or output_tokens < 1):
            raise ValueError("output_tokens must be a positive integer or None")
        self.heads = config.attention_heads
        self.head_dim = config.model_dim // self.heads
        self.refine_latents = refine_latents
        self.output_tokens = output_tokens
        width = config.model_dim
        self.latent_norm = RMSNorm(width)
        self.token_norm = RMSNorm(width)
        self.latent_reference = Linear(width, width, bias=False)
        self.latent_value = Linear(width, width, bias=False)
        self.token_reference = Linear(width, width, bias=False)
        self.token_value = Linear(width, width, bias=False) if refine_latents else None
        self.token_output = Linear(width, width, bias=False)
        self.latent_output = Linear(width, width, bias=False) if refine_latents else None
        if config.zero_init_branches:
            nn.init.zeros_(self.token_output.weight)
            if self.latent_output is not None:
                nn.init.zeros_(self.latent_output.weight)
        self.token_attention_gate = GatedResidual(width)
        self.token_ffn_norm = RMSNorm(width)
        self.token_ffn = FusedFeedForward(config) if config.fused_mlp else FeedForward(config)
        self.token_ffn_gate = GatedResidual(width)
        self.latent_attention_gate = GatedResidual(width) if refine_latents else None
        self.latent_ffn_norm = RMSNorm(width) if refine_latents else None
        self.latent_ffn = (
            (FusedFeedForward(config) if config.fused_mlp else FeedForward(config))
            if refine_latents
            else None
        )
        self.latent_ffn_gate = GatedResidual(width) if refine_latents else None
        self.latent_self_attention = Block(config) if refine_latents else None

    def _heads(self, values: Tensor) -> Tensor:
        batch, tokens, _width = values.shape
        return values.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, values: Tensor) -> Tensor:
        return values.transpose(1, 2).flatten(-2)

    def forward(
        self, latents: Tensor, tokens: Tensor, token_valid: Tensor
    ) -> tuple[Tensor, Tensor]:
        if self.output_tokens is not None and self.output_tokens > tokens.shape[1]:
            raise ValueError("output_tokens exceeds the input token count")
        tokens = torch.where(token_valid.unsqueeze(-1), tokens, 0.0)
        latent_input = self.latent_norm(latents)
        token_input = self.token_norm(tokens)
        latent_references = self._heads(self.latent_reference(latent_input))
        latent_values = self._heads(self.latent_value(latent_input))
        token_references = self._heads(self.token_reference(token_input))
        if self.refine_latents:
            assert self.token_value is not None
            # Bound training scratch independently of AOT's round-rematerialization
            # schedule. Dense compiled inference keeps vmapped rollout efficient.
            attention = (
                bounded_shared_score_attention
                if torch.is_grad_enabled()
                else shared_score_attention
            )
            latent_update, token_update = attention(
                latent_references,
                token_references,
                latent_values,
                self._heads(self.token_value(token_input)),
                token_valid,
                output_tokens=self.output_tokens,
            )
        else:
            latent_update = None
            token_update = _token_read(
                latent_references, token_references[:, :, : self.output_tokens], latent_values
            )
        tokens = tokens[:, : self.output_tokens]
        token_valid = token_valid[:, : self.output_tokens]
        tokens = self.token_attention_gate(
            tokens, self.token_output(self._merge_heads(token_update))
        )
        tokens = self.token_ffn_gate(tokens, self.token_ffn(self.token_ffn_norm(tokens)))
        tokens = torch.where(token_valid.unsqueeze(-1), tokens, 0.0)
        if self.refine_latents:
            assert latent_update is not None and self.latent_output is not None
            assert self.latent_attention_gate is not None and self.latent_ffn_norm is not None
            assert self.latent_ffn is not None and self.latent_ffn_gate is not None
            assert self.latent_self_attention is not None
            latents = self.latent_attention_gate(
                latents, self.latent_output(self._merge_heads(latent_update))
            )
            latents = self.latent_ffn_gate(latents, self.latent_ffn(self.latent_ffn_norm(latents)))
            latents = self.latent_self_attention(latents)
        return latents, tokens
