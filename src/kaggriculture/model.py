"""Entity-transformer policy and centralized distributional critic."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kaggriculture.actions import (
    N_MARKET_KINDS,
    N_QUANTITIES,
    N_UNIT_ACTIONS,
    MarketKind,
    UnitAction,
)
from kaggriculture.constants import (
    BOARD_SIZE,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    QUANTITY_BINS,
)
from kaggriculture.encoding import (
    BOARD_CHANNELS,
    CRITIC_FEATURES,
    GLOBAL_FEATURES,
    UNIT_FEATURES,
)

#: `modded-nanogpt` never lets its readout logits grow without bound: the
#: cross-entropy reads `23 * sigmoid((z + 5) / 7.5)` (`train_gpt.py:1690`,
#: `triton_kernels.py:1208-1213`), so however far `lm_head` drifts, the loss
#: gradient it produces is bounded and its derivative decays. This trainer's
#: categorical value head is zero-initialized, carries the only elevated Adam
#: rate in the critic, and fed an unbounded logit into HL-Gauss CE; measured
#: across three controlled 100-wave runs its weight norm was the one
#: exponentially growing parameter, at a rate no optimizer change moved
#: (RUNS.md, "The invariant driver"). The constants are the reference's.
#:
#: The cap costs nothing in expressiveness here. An HL-Gauss target with
#: `sigma = 0.75` bin widths puts essentially all of its mass inside four
#: bins, where the optimal logit span is about eight nats; 23 covers it with
#: room, and the atoms outside contribute `exp(-23)` of the partition sum.
VALUE_LOGIT_SOFTCAP = 23.0
VALUE_LOGIT_SOFTCAP_SHIFT = 5.0
VALUE_LOGIT_SOFTCAP_WIDTH = 7.5


def softcap_value_logits(logits: Tensor) -> Tensor:
    """Bound categorical logits in FP32 without erasing small readout differences."""

    logits = logits.float()
    return VALUE_LOGIT_SOFTCAP * torch.sigmoid(
        (logits + VALUE_LOGIT_SOFTCAP_SHIFT) / VALUE_LOGIT_SOFTCAP_WIDTH
    )


def categorical_value_support(minimum: float, maximum: float, atoms: int) -> Tensor:
    """Build linear buckets by mirroring one half, with an exact odd midpoint."""
    midpoint = (minimum + maximum) * 0.5
    half = atoms // 2
    step = (maximum - minimum) / (atoms - 1)
    left_end = midpoint if atoms % 2 else midpoint - step * 0.5
    left = torch.linspace(minimum, left_end, half + atoms % 2, dtype=torch.float32)
    right = minimum + maximum - left[:half].flip(-1)
    return torch.cat((left, right))


def _symmetric_sum(values: Tensor) -> Tensor:
    """Reduce mirrored pairs first so reflection cannot change the normalizer."""
    half = values.shape[-1] // 2
    paired = values[..., :half] + values[..., -half:].flip(-1)
    result = paired.sum(dim=-1)
    return result + values[..., half] if values.shape[-1] % 2 else result


def categorical_value(logits: Tensor, support: Tensor) -> Tensor:
    """Read a mirrored support without cancellation bias at symmetric logits."""
    probabilities = logits.float().softmax(dim=-1)
    support_float = support.float()
    half = support.numel() // 2
    midpoint = (support_float[0] + support_float[-1]) * 0.5
    paired = probabilities[..., -half:] - probabilities[..., :half].flip(-1)
    return midpoint + (paired * (support_float[-half:] - midpoint)).sum(dim=-1)


@dataclass(frozen=True)
class ModelConfig:
    cnn_width: int = 48
    cnn_blocks: int = 2
    model_dim: int = 96
    transformer_layers: int = 7
    attention_heads: int = 4
    ffn_multiplier: int = 4
    quantity_rank: int = 32
    value_atoms: int = 255
    value_min: float = -2.2
    value_max: float = 2.2
    value_sigma_ratio: float = 3.0
    # HL-Gauss is the default. The scalar ablation uses one output and half
    # squared error with no target or prediction clipping; the four value-support
    # fields above are inert in that mode.
    scalar_value: bool = False

    def __post_init__(self) -> None:
        if self.cnn_width <= 0:
            raise ValueError("cnn_width must be positive")
        if self.cnn_blocks <= 0:
            raise ValueError("cnn_blocks must be positive")
        if self.model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if self.transformer_layers < 3 or self.transformer_layers % 2 == 0:
            raise ValueError("transformer_layers must be odd and at least 3")
        if self.attention_heads <= 0 or self.model_dim % self.attention_heads:
            raise ValueError("attention_heads must evenly divide model_dim")
        if (self.model_dim // self.attention_heads) % 4:
            raise ValueError("attention head width must be divisible by 4 for axial RoPE")
        if self.ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be positive")
        if self.quantity_rank <= 0:
            raise ValueError("quantity_rank must be positive")
        if self.value_atoms < 2:
            raise ValueError("value_atoms must be at least 2")
        if not math.isfinite(self.value_min) or not math.isfinite(self.value_max):
            raise ValueError("value support bounds must be finite")
        if self.value_min >= self.value_max:
            raise ValueError("value_min must be smaller than value_max")
        if not math.isfinite(self.value_sigma_ratio) or self.value_sigma_ratio <= 0:
            raise ValueError("value_sigma_ratio must be finite and positive")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class ActorOutput(NamedTuple):
    unit_logits: Tensor
    market_kind_logits: Tensor
    market_quantity_context: Tensor


class BeliefOutput(NamedTuple):
    """Policy heads plus the belief latent the NextLat auxiliary supervises.

    Deliberately not a fourth field on `ActorOutput`. The native rollout's packed
    device-to-host transfer, the CUDA-graph capture of the collection forward and
    the vmapped frozen ensemble all consume that tuple whole -- `torch.vmap`
    outright rejects a `None` in its output pytree -- and none of them has any use
    for a latent only the update path reads. Keeping `ActorOutput` at three
    tensors leaves every collection path structurally untouched.
    """

    output: ActorOutput
    #: [rows, MAX_UNITS + MAX_MARKET_ORDERS, model_dim], fp32. The unit tokens come
    #: first, so the consumer splits at `MAX_UNITS` to recover the two halves.
    belief: Tensor


def _group_count(width: int) -> int:
    if width <= 0:
        raise ValueError("model width must be positive")
    groups = min(8, width)
    while width % groups:
        groups -= 1
    return groups


def policy_compile_options(mode: str) -> dict[str, Any]:
    """Preserve policy arithmetic across inference and training specializations.

    Reduced BF16 accumulation and inference-only graph rewrites can change a
    stored behavior likelihood before any optimizer step. Keep tensor-core
    BF16 compute, but retain explicit cast boundaries and the chosen operators.
    The cuBLAS reduction policy is process-wide, as required by PyTorch.
    """
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return {
        **torch._inductor.list_mode_options(mode),
        "emulate_precision_casts": True,
        "pattern_matcher": False,
    }


class Linear(nn.Linear):
    """Keep the bias epilogue identical across rollout and update batch sizes.

    A fused BF16 addmm can round the accumulator before or after adding bias,
    depending on the selected GEMM. Make that boundary explicit while retaining
    tensor-core matrix multiplication and FP32 optimizer master parameters.
    """

    def forward(self, inputs: Tensor) -> Tensor:
        output = F.linear(inputs, self.weight, None)
        return output if self.bias is None else output + self.bias.to(output.dtype)


class RMSNorm(nn.RMSNorm):
    """Compute-dtype normalization with fixed CUDA forward reduction arithmetic.

    `scale`/`shift` of shape `(*inputs.shape[:-2], width)` apply the conditioning
    `(1 + scale) * normalized + shift`, broadcast over the token dimension. On
    CUDA the epilogue is fused into the normalization kernel.
    """

    def forward(
        self, inputs: Tensor, scale: Tensor | None = None, shift: Tensor | None = None
    ) -> Tensor:
        if inputs.is_cuda:
            from kaggriculture.triton_norm import rms_norm

            eps = self.eps if self.eps is not None else torch.finfo(torch.float32).eps
            return rms_norm(inputs, self.weight, eps, scale, shift)
        with torch.autocast(inputs.device.type, enabled=False):
            normalized = F.rms_norm(inputs, self.normalized_shape, self.weight, self.eps)
        if scale is None or shift is None:
            return normalized
        return normalized * (1 + scale.unsqueeze(-2)) + shift.unsqueeze(-2)


class ReluSquared(nn.Module):
    """Parameter-free ReLU-squared activation used throughout the network.

    Written as a self-multiply rather than `.square()` because `aten::pow` sits on
    autocast's fp32 cast list while `aten::mul` does not. Under the production
    bf16 autocast `.square()` therefore upcast its input, ran in fp32, and returned
    fp32 -- on the FFN's 4x-wide activation, the widest tensor in the model. The
    precision bought nothing: every consumer is a `Linear` or `Conv2d`, which
    autocast casts straight back down, so both spellings round exactly once. The
    self-multiply keeps the activation in the compute dtype and halves its traffic.
    """

    def forward(self, inputs: Tensor) -> Tensor:
        activated = F.relu(inputs)
        return activated * activated


class ResidualBlock(nn.Module):
    """Pre-activation residual block used inside each spatial U-Net stage."""

    def __init__(self, width: int) -> None:
        super().__init__()
        groups = _group_count(width)
        self.layers = nn.Sequential(
            nn.GroupNorm(groups, width),
            ReluSquared(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(groups, width),
            ReluSquared(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.layers(inputs)


def _linear_upsample_weights(input_size: int, output_size: int) -> Tensor:
    """Interpolation matrix equal to align_corners=False linear resampling.

    Derived by resampling the identity through F.interpolate itself, so the
    weights match ATen's boundary handling exactly rather than re-deriving it.
    """
    identity = torch.eye(input_size).unsqueeze(1)
    resampled = F.interpolate(identity, size=output_size, mode="linear", align_corners=False)
    return resampled.squeeze(1).T.contiguous()


class SpatialUNet(nn.Module):
    """A compact two-resolution U-Net that retains one token per board tile."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        width = config.cnn_width
        low_width = 2 * width
        self.input = nn.Conv2d(BOARD_CHANNELS, width, 3, padding=1)
        self.encoder = nn.Sequential(*(ResidualBlock(width) for _ in range(config.cnn_blocks)))
        self.down = nn.Sequential(
            nn.GroupNorm(_group_count(width), width),
            ReluSquared(),
            nn.Conv2d(width, low_width, 3, stride=2, padding=1, bias=False),
        )
        self.bottleneck = nn.Sequential(
            *(ResidualBlock(low_width) for _ in range(config.cnn_blocks))
        )
        self.up_projection = nn.Sequential(
            nn.GroupNorm(_group_count(low_width), low_width),
            ReluSquared(),
            nn.Conv2d(low_width, width, 3, padding=1, bias=False),
        )
        self.decoder = nn.Sequential(*(ResidualBlock(width) for _ in range(config.cnn_blocks)))
        self.output = nn.Sequential(
            nn.GroupNorm(_group_count(width), width),
            ReluSquared(),
            nn.Conv2d(width, config.model_dim, 1),
        )
        low_size = (BOARD_SIZE + 1) // 2
        self.register_buffer(
            "upsample_weights", _linear_upsample_weights(low_size, BOARD_SIZE), persistent=False
        )

    def forward(self, board: Tensor) -> Tensor:
        high_resolution = self.encoder(self.input(board))
        low_resolution = self.bottleneck(self.down(high_resolution))
        # Bilinear upsampling as two separable matmuls. ATen's NCHW
        # upsample_bilinear2d kernel parallelizes only over output pixels and is
        # two orders of magnitude slower than these GEMMs at this shape. The
        # matmul form matches F.interpolate to fp32 rounding (~5e-7) at
        # "highest" matmul precision; under the deployed TF32 setting the GEMM
        # rounds to ~2e-3 on unit-scale activations, the same order as the
        # TF32 noise the network's other matmuls already carry.
        upsampled = self.upsample_weights @ low_resolution @ self.upsample_weights.T
        decoded = (self.up_projection(upsampled) + high_resolution) * math.sqrt(0.5)
        return self.output(self.decoder(decoded))


class AxialRotaryEmbedding(nn.Module):
    """Real-valued, compile-friendly 2D RoPE for arbitrary board coordinates."""

    def __init__(
        self,
        head_dim: int,
        board_size: int = BOARD_SIZE,
        theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        if head_dim % 4:
            raise ValueError("axial RoPE head width must be divisible by 4")
        frequencies = 1.0 / (
            theta ** (torch.arange(0, head_dim, 4, dtype=torch.float32) / head_dim)
        )
        axis = torch.arange(board_size, dtype=torch.float32)
        axis_angles = torch.outer(axis, frequencies)
        axis_cos = axis_angles.cos().repeat_interleave(2, dim=-1)
        axis_sin = axis_angles.sin().repeat_interleave(2, dim=-1)
        y, x = torch.meshgrid(
            torch.arange(board_size),
            torch.arange(board_size),
            indexing="ij",
        )
        flat_x = x.reshape(-1)
        flat_y = y.reshape(-1)
        self.board_size = board_size
        self.register_buffer(
            "cosine",
            torch.cat((axis_cos[flat_x], axis_cos[flat_y]), dim=-1),
            persistent=False,
        )
        self.register_buffer(
            "sine",
            torch.cat((axis_sin[flat_x], axis_sin[flat_y]), dim=-1),
            persistent=False,
        )

    @staticmethod
    def _rotate_pairs(inputs: Tensor) -> Tensor:
        pairs = inputs.unflatten(-1, (-1, 2))
        real, imaginary = pairs.unbind(dim=-1)
        return torch.stack((-imaginary, real), dim=-1).flatten(-2)

    def rotation(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Gather per-token cos/sin tables once for reuse by every layer.

        The gather output depends only on token positions, which are identical
        across all transformer layers, so hoisting it out of the attention
        modules removes the layer-count multiple of both the gather kernels and
        their saved activations.
        """
        if positions.ndim != 3 or positions.shape[-1] != 2:
            raise ValueError("RoPE positions must have shape [batch, tokens, 2]")
        coordinates = positions.long().clamp(0, self.board_size - 1)
        indices = coordinates[..., 1] * self.board_size + coordinates[..., 0]
        batch, tokens = indices.shape
        cosine = self.cosine.index_select(0, indices.reshape(-1)).view(batch, 1, tokens, -1)
        sine = self.sine.index_select(0, indices.reshape(-1)).view_as(cosine)
        return cosine, sine

    @classmethod
    def apply_rotation(
        cls, query: Tensor, key: Tensor, rotation: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        if query.shape != key.shape:
            raise ValueError("query and key shapes must match for self-attention RoPE")
        cosine, sine = rotation
        if cosine.shape[0] != query.shape[0] or cosine.shape[-2:] != query.shape[-2:]:
            raise ValueError("RoPE rotation does not match the query batch or token layout")
        cosine = cosine.to(dtype=query.dtype)
        sine = sine.to(dtype=query.dtype)
        return (
            query * cosine + cls._rotate_pairs(query) * sine,
            key * cosine + cls._rotate_pairs(key) * sine,
        )


def _sdpa_inputs(query: Tensor, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Use BF16 only inside CUDA SDPA so FP32 PPO heads remain replay-stable."""
    if query.device.type == "cuda":
        # CUDA Flash SDPA is generally unavailable for FP32. Keeping this cast
        # local avoids changing the CNN, residual stream, or policy head dtype.
        return query.bfloat16(), key.bfloat16(), value.bfloat16()
    return query, key, value


class SelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.attention_heads
        self.head_dim = config.model_dim // self.heads
        self.qkv = Linear(config.model_dim, 3 * config.model_dim, bias=False)
        self.query_norm = RMSNorm(self.head_dim)
        self.key_norm = RMSNorm(self.head_dim)
        self.output = Linear(config.model_dim, config.model_dim, bias=False)

    def forward(
        self,
        inputs: Tensor,
        rotation: tuple[Tensor, Tensor],
        readout: int | None = None,
    ) -> Tensor:
        batch, tokens, width = inputs.shape
        if readout is not None and not 0 < readout <= tokens:
            raise ValueError("readout must name a nonempty prefix of the tokens")
        qkv = self.qkv(inputs).view(batch, tokens, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        query = self.query_norm(query)
        key = self.key_norm(key)
        query, key = AxialRotaryEmbedding.apply_rotation(query, key, rotation)
        query_tokens = tokens
        if readout is not None:
            # Every token stays a key and a value; only the leading `readout` rows
            # are still queried, and the slice sits after the rotation, so the kept
            # rows are algebraically the same computation. They are not bit-equal
            # on CUDA: a one-row query selects a different Flash tiling and
            # `_sdpa_inputs` has already dropped to bf16, so the kept rows move by
            # roughly one bf16 rounding. Only the critic narrows, and its value
            # estimate carries no cross-path parity contract.
            query = query[:, :, :readout]
            query_tokens = readout
        query, key, value = _sdpa_inputs(query, key, value)
        # Inactive unit tokens are explicitly zeroed at every block boundary.
        # Omitting an attention mask keeps this static-shape call Flash-eligible.
        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        attended = attended.to(dtype=inputs.dtype)
        attended = attended.transpose(1, 2).reshape(batch, query_tokens, width)
        return self.output(attended)


class ReluSquaredFeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.model_dim * config.ffn_multiplier
        self.input = Linear(config.model_dim, hidden)
        self.activation = ReluSquared()
        self.output = Linear(hidden, config.model_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.output(self.activation(self.input(inputs)))


class TransformerBlock(nn.Module):
    """Pre-norm attention and MLP block with deterministic residuals."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.model_dim)
        self.attention = SelfAttention(config)
        self.ffn_norm = RMSNorm(config.model_dim)
        self.ffn = ReluSquaredFeedForward(config)

    def forward(
        self,
        inputs: Tensor,
        rotation: tuple[Tensor, Tensor],
        valid: Tensor | None,
        readout: int | None = None,
    ) -> Tensor:
        if valid is None:
            attended = self.attention(self.attention_norm(inputs), rotation, readout)
            hidden = (inputs if readout is None else inputs[:, :readout]) + attended
            return hidden + self.ffn(self.ffn_norm(hidden))
        if readout is not None:
            raise ValueError("a masked block cannot drop query rows")
        hidden = torch.where(valid, inputs, 0.0)
        hidden = torch.where(
            valid,
            hidden + self.attention(self.attention_norm(hidden), rotation),
            0.0,
        )
        return torch.where(valid, hidden + self.ffn(self.ffn_norm(hidden)), 0.0)


class EntityTransformer(nn.Module):
    """Odd-depth transformer with mirrored encoder-to-decoder residual skips."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        side_depth = config.transformer_layers // 2
        self.rope = AxialRotaryEmbedding(config.model_dim // config.attention_heads)
        self.encoder = nn.ModuleList(TransformerBlock(config) for _ in range(side_depth))
        self.bottleneck = TransformerBlock(config)
        self.decoder = nn.ModuleList(TransformerBlock(config) for _ in range(side_depth))
        self.output_norm = RMSNorm(config.model_dim)

    def forward(
        self,
        inputs: Tensor,
        positions: Tensor,
        valid: Tensor | None,
        readout: int | None = None,
    ) -> Tensor:
        """Run the trunk, optionally narrowing the final block to a token prefix.

        `valid` gates the residual stream rather than attention itself, so an
        all-true mask is arithmetically the identity. Passing `None` says so and
        drops 3 full-tensor `torch.where` per block plus 2 more here -- 26 masked
        writes over the critic's [B, 101, 96] stream that provably changed nothing.
        Callers whose mask is genuinely mixed must keep passing it: invalid rows are
        still attended to as zero-valued keys, so zeroing them is load-bearing.

        `readout` says the caller reads only the leading rows of the result. The last
        block then queries just those, while every token still supplies a key and a
        value, so the returned rows are the same computation up to the attention
        backend's tiling. A consumer reading one row of 101 was paying for the final
        block's output projection and 4x-wide FFN on a hundred token rows that
        nothing downstream could observe; those rows already received no gradient,
        since the value head discarded them.
        """
        rotation = self.rope.rotation(positions)
        hidden = inputs if valid is None else torch.where(valid, inputs, 0.0)
        skips: list[Tensor] = []
        for block in self.encoder:
            hidden = block(hidden, rotation, valid)
            skips.append(hidden)
        hidden = self.bottleneck(hidden, rotation, valid)
        last = len(self.decoder) - 1
        for index, (block, skip) in enumerate(zip(self.decoder, reversed(skips), strict=True)):
            merged = (hidden + skip) * math.sqrt(0.5)
            hidden = merged if valid is None else torch.where(valid, merged, 0.0)
            hidden = block(hidden, rotation, valid, readout if index == last else None)
        normalized = self.output_norm(hidden)
        return normalized if valid is None else torch.where(valid, normalized, 0.0)


def _board_positions() -> Tensor:
    y, x = torch.meshgrid(
        torch.arange(BOARD_SIZE),
        torch.arange(BOARD_SIZE),
        indexing="ij",
    )
    return torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)


def initialize_policy_heads(
    unit_head: nn.Linear,
    market_kind: nn.Linear,
    market_quantity_context: nn.Linear,
    market_quantity_kind_gate: nn.Embedding,
    market_quantity_value: nn.Embedding,
    market_quantity_bias: nn.Parameter,
) -> None:
    """Initialize the shared policy heads every actor architecture uses."""
    for head in (unit_head, market_kind, market_quantity_context):
        nn.init.normal_(head.weight, std=0.01)
        if head.bias is not None:
            nn.init.zeros_(head.bias)
    nn.init.zeros_(market_quantity_kind_gate.weight)
    nn.init.normal_(market_quantity_value.weight, std=0.01)
    nn.init.zeros_(market_quantity_bias)

    # Legal masks already remove actions that cannot have an effect. Among
    # the remaining actions, favor completing an economic cycle over random
    # movement or destroying an investment. Every action remains learnable.
    with torch.no_grad():
        unit_bias = unit_head.bias
        unit_bias[UnitAction.PASS] = -1.25
        unit_bias[UnitAction.DROP] = 2.0
        for item, maximum, offset in (
            ("WHEAT", 16, 1.0),
            ("FERTILIZER", 8, 0.75),
            ("GOOSE", 4, 0.75),
            ("COW", 4, 0.75),
            ("SHEEP", 4, 0.75),
        ):
            for quantity in range(1, maximum + 1):
                unit_bias[UnitAction[f"PICKUP_{item}_{quantity}"]] = offset - math.log(quantity)
        unit_bias[UnitAction.PLACE_GOOSE : UnitAction.PLACE_SHEEP + 1] = 2.0
        unit_bias[UnitAction.PLANT_WHEAT : UnitAction.PLANT_MELON + 1] = 1.0
        unit_bias[UnitAction.WATER] = 2.0
        unit_bias[UnitAction.HARVEST] = 2.5
        unit_bias[UnitAction.FERTILIZE] = 1.0
        unit_bias[UnitAction.DIG] = -0.5
        unit_bias[UnitAction.BUILD_COOP : UnitAction.BUILD_PASTURE + 1] = -2.5
        unit_bias[UnitAction.FEED] = 2.0
        unit_bias[UnitAction.COLLECT_FERTILIZER] = 2.0
        unit_bias[UnitAction.CARE] = 1.0

        # Keep roughly 95% opening STOP probability and bias initial
        # exploration toward cheap hires/seeds and inventory liquidation.
        kind_bias = market_kind.bias
        kind_bias[MarketKind.STOP] = 4.5
        kind_bias[MarketKind.HIRE] = 1.0
        kind_bias[MarketKind.BUY_LAND] = -7.0
        kind_bias[MarketKind.BUY_SEED_WHEAT : MarketKind.BUY_SEED_MELON + 1] = -1.0
        kind_bias[MarketKind.BUY_PRODUCT_WHEAT : MarketKind.BUY_PRODUCT_FERTILIZER + 1] = -3.0
        kind_bias[MarketKind.BUY_ANIMAL_GOOSE : MarketKind.BUY_ANIMAL_SHEEP + 1] = -4.0
        kind_bias[MarketKind.SELL_WHEAT : MarketKind.SELL_FERTILIZER + 1] = 4.0

        quantities = torch.as_tensor(
            QUANTITY_BINS,
            device=market_quantity_bias.device,
            dtype=market_quantity_bias.dtype,
        )
        log_quantity = quantities.log()
        market_quantity_bias[MarketKind.BUY_SEED_WHEAT : MarketKind.BUY_SEED_MELON + 1].copy_(
            -2.0 * log_quantity
        )
        market_quantity_bias[MarketKind.BUY_PRODUCT_WHEAT : MarketKind.BUY_ANIMAL_SHEEP + 1].copy_(
            -2.5 * log_quantity
        )
        market_quantity_bias[MarketKind.SELL_WHEAT : MarketKind.SELL_FERTILIZER + 1].copy_(
            0.5 * log_quantity
        )


def factored_quantity_logits(
    quantity_context: Tensor,
    market_kinds: Tensor,
    kind_gate: nn.Embedding,
    quantity_value: nn.Embedding,
    quantity_bias: Tensor,
    quantity_rank: int,
) -> Tensor:
    """Score the native sampler's small FP32 head, even under trunk autocast."""
    if quantity_context.shape[:-1] != market_kinds.shape:
        raise ValueError("quantity context and selected market kinds must align")
    if quantity_context.shape[-1] != quantity_rank:
        raise ValueError("quantity context has the wrong feature width")
    with torch.autocast(quantity_context.device.type, enabled=False):
        kinds = market_kinds.long()
        quantity_features = quantity_context.float() * (1.0 + kind_gate(kinds).float())
        # A matmul here may use TF32 under the trainer's global "high"
        # setting, even with autocast disabled. This small rank reduction is
        # fused by Inductor and follows the Rust sampler's FP32 accumulation.
        values = quantity_value.weight.float()
        scores = quantity_bias[kinds].float()
        for rank in range(quantity_rank):
            scores = scores + quantity_features[..., rank, None] * values[:, rank]
        return scores


class FarmActor(nn.Module):
    """Decentralized entity policy using only the acting player's private state."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.config = config
        self.spatial = SpatialUNet(config)
        self.state_projection = Linear(GLOBAL_FEATURES, config.model_dim)
        self.unit_projection = Linear(UNIT_FEATURES, config.model_dim, bias=False)
        self.unit_slots = nn.Embedding(MAX_UNITS, config.model_dim)
        self.market_queries = nn.Embedding(MAX_MARKET_ORDERS, config.model_dim)
        self.token_types = nn.Embedding(4, config.model_dim)
        self.transformer = EntityTransformer(config)
        self.unit_head = nn.Sequential(
            RMSNorm(config.model_dim),
            Linear(config.model_dim, N_UNIT_ACTIONS),
        )
        self.market_norm = RMSNorm(config.model_dim)
        self.market_kind = Linear(config.model_dim, N_MARKET_KINDS)
        # A dense model_dim -> kind x exact-quantity head would be a material
        # fraction of the policy. This state x kind factorization retains a
        # learned interaction plus a fully expressive kind/quantity bias.
        self.market_quantity_context = Linear(config.model_dim, config.quantity_rank, bias=False)
        self.market_quantity_kind_gate = nn.Embedding(N_MARKET_KINDS, config.quantity_rank)
        self.market_quantity_value = nn.Embedding(N_QUANTITIES, config.quantity_rank)
        self.market_quantity_bias = nn.Parameter(torch.zeros(N_MARKET_KINDS, N_QUANTITIES))
        self.register_buffer("board_positions", _board_positions(), persistent=False)
        self._initialize_policy_heads()

    def _initialize_policy_heads(self) -> None:
        initialize_policy_heads(
            self.unit_head[-1],
            self.market_kind,
            self.market_quantity_context,
            self.market_quantity_kind_gate,
            self.market_quantity_value,
            self.market_quantity_bias,
        )

    def quantity_logits(self, quantity_context: Tensor, market_kinds: Tensor) -> Tensor:
        """Score exact quantities only for the already-selected market kind."""
        return factored_quantity_logits(
            quantity_context,
            market_kinds,
            self.market_quantity_kind_gate,
            self.market_quantity_value,
            self.market_quantity_bias,
            self.config.quantity_rank,
        )

    def _head_inputs(
        self,
        board: Tensor,
        global_features: Tensor,
        units: Tensor,
        unit_positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run the trunk and return the two tensors the policy heads are applied to.

        Split out so `forward` and `forward_with_belief` share one trunk pass without
        `forward` paying for the belief. Delegating `forward` to `forward_with_belief`
        would have put the belief's concatenation and its two dtype widens on the
        collection forward, which `_packed_outputs_to_host` records as running 719
        times per training iteration. Nothing downstream could observe the result, but
        `eager` is a live rollout mode and the one every CPU test takes, so the
        allocation would be real rather than something Inductor deletes. Only the
        update reads a belief; it should be the only caller that allocates one.
        """
        batch = board.shape[0]
        board_tokens = self.spatial(board).flatten(2).transpose(1, 2)
        state_token = self.state_projection(global_features).unsqueeze(1)
        unit_valid = units[..., :1].gt(0.5)
        masked_units = torch.where(unit_valid, units, 0.0)
        unit_tokens = self.unit_projection(masked_units)
        unit_tokens = unit_tokens + self.unit_slots.weight.unsqueeze(0)
        market_tokens = self.market_queries.weight.unsqueeze(0).expand(batch, -1, -1)

        state_token = state_token + self.token_types.weight[0]
        board_tokens = board_tokens + self.token_types.weight[1]
        unit_tokens = unit_tokens + self.token_types.weight[2]
        market_tokens = market_tokens + self.token_types.weight[3]
        tokens = torch.cat((state_token, board_tokens, unit_tokens, market_tokens), dim=1)

        fixed_valid = torch.ones(
            batch,
            1 + BOARD_SIZE * BOARD_SIZE,
            1,
            device=tokens.device,
            dtype=torch.bool,
        )
        market_valid = torch.ones(
            batch,
            MAX_MARKET_ORDERS,
            1,
            device=tokens.device,
            dtype=torch.bool,
        )
        valid = torch.cat((fixed_valid, unit_valid, market_valid), dim=1)

        static_positions = self.board_positions.unsqueeze(0).expand(batch, -1, -1)
        zero_position = torch.zeros(
            batch,
            1,
            2,
            device=unit_positions.device,
            dtype=unit_positions.dtype,
        )
        market_positions = zero_position.expand(batch, MAX_MARKET_ORDERS, 2)
        positions = torch.cat(
            (zero_position, static_positions, unit_positions, market_positions), dim=1
        )
        hidden = self.transformer(tokens, positions, valid)

        unit_start = 1 + BOARD_SIZE * BOARD_SIZE
        market_start = unit_start + MAX_UNITS
        return hidden[:, unit_start:market_start], self.market_norm(hidden[:, market_start:])

    def _policy_heads(self, unit_hidden: Tensor, market_hidden: Tensor) -> ActorOutput:
        """Score every action factor from the trunk tensors `_head_inputs` returned."""
        return ActorOutput(
            unit_logits=self.unit_head(unit_hidden).contiguous(),
            market_kind_logits=self.market_kind(market_hidden).contiguous(),
            market_quantity_context=self.market_quantity_context(market_hidden).contiguous(),
        )

    def forward(
        self,
        board: Tensor,
        global_features: Tensor,
        units: Tensor,
        unit_positions: Tensor,
    ) -> ActorOutput:
        return self._policy_heads(*self._head_inputs(board, global_features, units, unit_positions))

    def forward_with_belief(
        self,
        board: Tensor,
        global_features: Tensor,
        units: Tensor,
        unit_positions: Tensor,
    ) -> BeliefOutput:
        """Score every action factor and expose the latent the heads read to do it.

        The belief is the heads' own input: `unit_hidden` followed by `market_hidden`,
        exactly the two tensors `_policy_heads` applies the three heads to. That
        coupling is the point. The reference regresses the same vector that produces
        the output, so predicting the next latent is predicting the next decision;
        supervising a token no head reads -- the trunk's state token being the obvious
        candidate -- is cheaply satisfiable instead, because the trunk can park an
        easily extrapolated function of the step in an unread channel and drive the
        auxiliary loss to zero without the policy changing at all.

        The two halves keep their own preprocessing: the unit slice is raw trunk
        output and the market slice has already been through `market_norm`, because
        that is what `unit_head` and `market_kind`/`market_quantity_context`
        respectively receive. The consumer splits the result back at `MAX_UNITS`.

        The cast to fp32 is load-bearing. `RMSNorm` deliberately returns the compute
        dtype, so under the production bf16 autocast these tensors are bf16, while the
        auxiliary's dynamics MLP holds fp32 parameters and must not silently take a
        bf16 input. Each half is widened before the concatenation rather than after,
        so the two slices cannot reach `cat` in different dtypes.
        """
        unit_hidden, market_hidden = self._head_inputs(
            board, global_features, units, unit_positions
        )
        return BeliefOutput(
            self._policy_heads(unit_hidden, market_hidden),
            torch.cat((unit_hidden.float(), market_hidden.float()), dim=1),
        )


class DistributionalCritic(nn.Module):
    """Separate centralized critic with both players' private state."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.config = config
        self.spatial = SpatialUNet(config)
        self.state_projection = Linear(CRITIC_FEATURES, config.model_dim)
        self.token_types = nn.Embedding(2, config.model_dim)
        self.transformer = EntityTransformer(config)
        self.value_head = Linear(config.model_dim, 1 if config.scalar_value else config.value_atoms)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        self.register_buffer("board_positions", _board_positions(), persistent=False)
        self.register_buffer(
            "support",
            categorical_value_support(config.value_min, config.value_max, config.value_atoms),
            persistent=True,
        )

    def forward(self, board: Tensor, critic_features: Tensor) -> Tensor:
        batch = board.shape[0]
        state_token = self.state_projection(critic_features).unsqueeze(1)
        state_token = state_token + self.token_types.weight[0]
        board_tokens = self.spatial(board).flatten(2).transpose(1, 2)
        board_tokens = board_tokens + self.token_types.weight[1]
        tokens = torch.cat((state_token, board_tokens), dim=1)
        zero_position = torch.zeros(
            batch,
            1,
            2,
            device=self.board_positions.device,
            dtype=self.board_positions.dtype,
        )
        positions = torch.cat(
            (zero_position, self.board_positions.unsqueeze(0).expand(batch, -1, -1)),
            dim=1,
        )
        # The state token and all hundred board tokens are always present, so this
        # critic has no invalid rows to mask, and the value is read from the state
        # token alone -- a single query into the final block.
        value_token = self.transformer(tokens, positions, None, readout=1)[:, 0]
        readout = self.value_head(value_token)
        if self.config.scalar_value:
            return readout.contiguous()
        return softcap_value_logits(readout).contiguous()

    def value(self, logits: Tensor) -> Tensor:
        if self.config.scalar_value:
            return logits.float().squeeze(-1)
        return categorical_value(logits, self.support)


def hl_gauss_value_targets(
    targets: Tensor,
    support: Tensor,
    sigma_ratio: float = ModelConfig.value_sigma_ratio,
    *,
    validate: bool = True,
) -> Tensor:
    """Integrate Gaussian label mass over categorical value bins."""
    if support.ndim != 1 or support.numel() < 2:
        raise ValueError("value support must be one-dimensional with at least two atoms")
    support_float = support.float()
    width = (support_float[-1] - support_float[0]) / (support.numel() - 1)
    if not math.isfinite(sigma_ratio) or sigma_ratio <= 0:
        raise ValueError("sigma_ratio must be finite and positive")

    targets_float = targets.float()
    if validate:
        if not bool(torch.isfinite(support_float).all()):
            raise ValueError("value support must be finite")
        widths = support_float[1:] - support_float[:-1]
        if not bool(torch.all(widths > 0)):
            raise ValueError("value support must be strictly increasing")
        rounding = 2 * torch.finfo(torch.float32).eps * support_float.abs().max()
        if not bool(torch.all((widths - width).abs() <= rounding)):
            raise ValueError("HL-Gauss requires evenly spaced value atoms")
        if not bool(torch.isfinite(targets_float).all()):
            raise ValueError("value targets must be finite")
        if bool(
            torch.any((targets_float < support_float[0]) | (targets_float > support_float[-1]))
        ):
            raise ValueError("value targets fall outside the critic support")

    midpoints = (support_float[1:] + support_float[:-1]) * 0.5
    edges = torch.cat(
        (support_float[:1] - width * 0.5, midpoints, support_float[-1:] + width * 0.5)
    )
    standardized = (edges - targets_float.unsqueeze(-1)) / (width * sigma_ratio * math.sqrt(2.0))
    lower, upper = standardized[..., :-1], standardized[..., 1:]
    erf = torch.erf(standardized)
    tails = torch.erfc(standardized.abs())
    central_mass = 0.5 * (erf[..., 1:] - erf[..., :-1])
    tail_mass = 0.5 * (tails[..., 1:] - tails[..., :-1]).abs()
    # erf differences preserve narrow central intervals; erfc preserves tails
    # that would vanish when two CDF values both round to one.
    central = ((lower <= 0) & (upper >= 0)) | ((lower.abs() < 1) & (upper.abs() < 1))
    probabilities = torch.where(central, central_mass, tail_mass).clamp_min(0.0)
    return probabilities / _symmetric_sum(probabilities).unsqueeze(-1)


def distributional_value_loss(
    logits: Tensor,
    targets: Tensor,
    support: Tensor,
    sigma_ratio: float = ModelConfig.value_sigma_ratio,
    *,
    validate: bool = True,
) -> Tensor:
    if logits.shape[:-1] != targets.shape or logits.shape[-1] != support.numel():
        raise ValueError("value logits, targets, and support shapes must align")
    projected = hl_gauss_value_targets(targets.detach(), support, sigma_ratio, validate=validate)
    return -(projected * logits.float().log_softmax(dim=-1)).sum(dim=-1)


def scalar_value_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """CleanRL's unclipped value loss: half the squared error, per state.

    `0.5 *` rather than a bare square because that is the reference's
    coefficient, and the critic's learning rate was tuned against a gradient of
    that size. Neither argument is clipped: this path exists to remove the
    bounded support, so re-introducing a bound on either the target or the
    prediction would defeat it.
    """
    if predictions.shape != targets.shape:
        raise ValueError(
            f"value predictions {tuple(predictions.shape)} must match targets "
            f"{tuple(targets.shape)}"
        )
    return 0.5 * (predictions.float() - targets.float()).square()


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
