"""Forward masking process for MDLM diffusion.

Given clean tokens x0, a PAD mask, timestep t, and a noise schedule,
stochastically masks non-PAD positions with probability (1 - alpha(t)).
Node positions are masked to NODE_MASK_IDX, edge positions to EDGE_MASK_IDX.

CRITICAL INVARIANT: PAD positions are NEVER masked. This is the most
important correctness property in the entire diffusion pipeline.
"""

from __future__ import annotations

import torch
from torch import Tensor

from bd_gen.data.vocab import (
    EDGE_MASK_IDX,
    NODE_MASK_IDX,
    RPLAN_VOCAB_CONFIG,
    VocabConfig,
)
from bd_gen.diffusion.noise_schedule import NoiseSchedule


def forward_mask(
    x0: Tensor,
    pad_mask: Tensor,
    t: Tensor,
    noise_schedule: NoiseSchedule,
    vocab_config: VocabConfig = RPLAN_VOCAB_CONFIG,
) -> tuple[Tensor, Tensor]:
    """Apply stochastic masking to clean tokens.

    For each non-PAD position, independently masks the token with probability
    (1 - alpha(t)). PAD positions are never touched.

    Args:
        x0: (B, SEQ_LEN) long tensor of clean token indices.
        pad_mask: (B, SEQ_LEN) bool tensor. True = real position.
        t: (B,) float32 tensor of timesteps in [0, 1].
        noise_schedule: NoiseSchedule providing alpha(t).
        vocab_config: VocabConfig for n_max (to distinguish node vs edge
            positions for correct MASK token). Defaults to RPLAN_VOCAB_CONFIG.

    Returns:
        Tuple of:
            x_t: (B, SEQ_LEN) long tensor with masked tokens.
            mask_indicators: (B, SEQ_LEN) bool tensor. True where the token
                WAS masked by this function. False for kept positions AND
                PAD positions.
    """
    B, SEQ_LEN = x0.shape
    n_max = vocab_config.n_max
    device = x0.device

    # Keeping probability alpha(t) per sample: (B,) -> (B, 1) for broadcasting
    alpha_t = noise_schedule.alpha(t).unsqueeze(1)  # (B, 1)

    # Draw uniform random per position; mask where rand >= alpha_t
    rand = torch.rand(B, SEQ_LEN, device=device)
    should_mask = rand >= alpha_t  # True where we WANT to mask (prob = 1 - alpha_t)

    # CRITICAL: only mask non-PAD positions
    should_mask = should_mask & pad_mask

    # Build MASK token tensor: node positions get NODE_MASK_IDX,
    # edge positions get EDGE_MASK_IDX
    mask_tokens = torch.full((SEQ_LEN,), NODE_MASK_IDX, dtype=torch.long, device=device)
    mask_tokens[n_max:] = EDGE_MASK_IDX
    mask_tokens = mask_tokens.unsqueeze(0).expand(B, -1)  # (B, SEQ_LEN)

    # Apply: where should_mask use mask_token, else keep x0
    x_t = torch.where(should_mask, mask_tokens, x0)

    return x_t, should_mask
