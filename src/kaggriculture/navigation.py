"""Stateless destination-based navigation over the existing engine actions.

Each noncurrent tile denotes the next step of a deterministic Manhattan route
(vertical before horizontal). We marginalize destination probabilities into
NORTH/SOUTH/EAST/WEST before sampling. Thus BC and PPO score the executed move,
not an unobserved destination, and the existing native sampler needs no changes.
The destination is reconsidered every turn: this is not route commitment.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from kaggriculture.constants import BOARD_SIZE
from kaggriculture.model import Linear, RMSNorm


def destination_directions(board_size: int = BOARD_SIZE) -> Tensor:
    """Map [origin, destination] to engine action 1..4; same tile maps to 0."""
    indices = torch.arange(board_size * board_size)
    y, x = indices // board_size, indices % board_size
    dy, dx = y[None, :] - y[:, None], x[None, :] - x[:, None]
    return torch.where(
        dy < 0, 1, torch.where(dy > 0, 2, torch.where(dx > 0, 3, torch.where(dx < 0, 4, 0)))
    )


def marginal_movement_logits(scores: Tensor, directions: Tensor, move_logit: Tensor) -> Tensor:
    """Exact log masses for four moves, with total mass exp(move_logit).

    Finite sentinels keep gradients defined for impossible edge directions;
    existing legality masks exclude those directions before the action softmax.
    """
    scores = scores.float()
    # Also finite after the final BF16 policy-logit boundary.
    floor = -1.0e9
    scores = scores.masked_fill(directions == 0, floor)
    normalizer = torch.logsumexp(scores, dim=-1)
    return torch.stack(
        [
            torch.logsumexp(scores.masked_fill(directions != action, floor), dim=-1)
            - normalizer
            + move_logit.float()
            for action in (1, 2, 3, 4)
        ],
        dim=-1,
    )


class TargetNavigation(nn.Module):
    """Learn a destination from unit and own-farm tile representations."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.unit_norm = RMSNorm(width)
        self.tile_norm = RMSNorm(width)
        self.query = Linear(width, width, bias=False)
        self.key = Linear(width, width, bias=False)
        self.move_gate = Linear(width, 1)
        self.scale = width**-0.5
        # A zero gate gives each legal direction unit prior mass.
        nn.init.zeros_(self.move_gate.weight)
        nn.init.zeros_(self.move_gate.bias)
        directions = destination_directions()
        counts = torch.stack([(directions == a).sum(-1) for a in (0, 1, 2, 3, 4)], dim=-1)
        # Correct unequal route-region sizes: uniform scores must not favor
        # vertical moves merely because vertical-first routing maps more tiles
        # to them. Within a next-step region the destination prior is uniform.
        prior = -counts.gather(-1, directions).clamp_min(1).float().log()
        legal_direction_count = (counts[:, 1:] > 0).sum(-1).float()
        self.register_buffer("directions", directions, persistent=False)
        self.register_buffer("destination_log_prior", prior, persistent=False)
        self.register_buffer("move_log_prior", legal_direction_count.log(), persistent=False)

    def forward(self, units: Tensor, tiles: Tensor, unit_categorical: Tensor) -> Tensor:
        units = self.unit_norm(units)
        tiles = self.tile_norm(tiles[:, : BOARD_SIZE * BOARD_SIZE])
        scores = torch.matmul(self.query(units), self.key(tiles).transpose(-1, -2)) * self.scale
        origins = unit_categorical[..., 2].long() * BOARD_SIZE + unit_categorical[..., 3].long()
        directions = self.directions[origins]
        return marginal_movement_logits(
            scores.float() + self.destination_log_prior[origins],
            directions,
            self.move_gate(units).squeeze(-1).float() + self.move_log_prior[origins],
        )


def assemble_unit_logits(local_logits: Tensor, movement_logits: Tensor) -> Tensor:
    """Insert navigation between PASS and local operations in engine order."""
    return torch.cat(
        (local_logits[..., :1].float(), movement_logits, local_logits[..., 1:].float()), dim=-1
    ).to(local_logits.dtype)
