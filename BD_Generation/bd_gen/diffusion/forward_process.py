"""Forward masking process for MDLM diffusion.

Given clean tokens x0, a PAD mask, timestep t, and a noise schedule,
stochastically masks non-PAD positions with probability (1 - alpha(t)).
Node positions are masked to NODE_MASK_IDX, edge positions to EDGE_MASK_IDX.

CRITICAL INVARIANT: PAD positions are NEVER masked. This is the most
important correctness property in the entire diffusion pipeline.
"""

from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn.functional as F
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


# =========================================================================
# v2 Learned Forward Process (STGS + per-position rates)
# =========================================================================


class STGSOutput(TypedDict):
    """Output of the learned forward masking process (training time).

    Contains both soft embeddings (for differentiable training through the
    Straight-Through Gumbel-Softmax) and hard discrete tokens (for the
    denoiser's standard forward pass and loss computation).
    """

    soft_embeddings: Tensor   # (B, SEQ_LEN, d_model) mixed clean+mask embeddings
    x_t: Tensor              # (B, SEQ_LEN) discrete masked tokens (hard decisions)
    mask_indicators: Tensor   # (B, SEQ_LEN) bool, True where masked
    alpha_per_pos: Tensor     # (B, SEQ_LEN) per-position alpha from rate network
    gumbel_weights: Tensor    # (B, SEQ_LEN, 2) soft keep/mask weights


def stgs_sample(alpha: Tensor, gumbel_temperature: float = 1.0) -> Tensor:
    """Straight-Through Gumbel-Softmax for discrete masking.

    Produces a differentiable approximation to discrete keep/mask decisions.
    In the forward pass, outputs are hard one-hot (exactly 0 or 1).
    In the backward pass, gradients flow through the soft Gumbel-Softmax.

    Args:
        alpha: (B, SEQ_LEN) per-position keeping probability in (0, 1).
        gumbel_temperature: Softmax temperature. Lower values produce
            harder (more peaked) distributions. Default 1.0.

    Returns:
        (B, SEQ_LEN, 2) tensor. Channel 0 = keep weight, channel 1 = mask
        weight. Forward: hard one-hot. Backward: soft Gumbel-Softmax gradient.
    """
    # Logits for 2-class categorical: [log(alpha), log(1-alpha)]
    logits = torch.stack([
        torch.log(alpha.clamp(min=1e-8)),
        torch.log((1 - alpha).clamp(min=1e-8)),
    ], dim=-1)  # (B, SEQ_LEN, 2)

    # Gumbel noise (float64 for precision, avoids log(0) issues)
    u = torch.rand_like(logits, dtype=torch.float64)
    u = u.clamp(min=1e-10, max=1 - 1e-10)
    gumbel = -torch.log(-torch.log(u))

    # Soft sample
    p_soft = F.softmax(
        (logits.double() + gumbel) / gumbel_temperature, dim=-1
    ).float()

    # Hard sample (argmax -> one-hot)
    hard_idx = p_soft.argmax(dim=-1)
    p_hard = F.one_hot(hard_idx, num_classes=2).float()

    # Straight-through: forward uses hard, backward uses soft gradient
    return p_hard - p_soft.detach() + p_soft


def forward_mask_learned(
    x0: Tensor,
    pad_mask: Tensor,
    t: Tensor,
    rate_network: torch.nn.Module,
    denoiser: torch.nn.Module,
    vocab_config: VocabConfig,
    gumbel_temperature: float = 1.0,
) -> STGSOutput:
    """Training-time forward masking with learned per-position rates and STGS.

    Uses Straight-Through Gumbel-Softmax to produce differentiable soft
    embeddings that mix clean and mask token embeddings, while also providing
    hard discrete tokens for loss computation.

    CRITICAL INVARIANT: PAD positions are NEVER masked. Gumbel weights for
    PAD positions are forced to keep=1, mask=0.

    Args:
        x0: (B, SEQ_LEN) long tensor of clean token indices.
        pad_mask: (B, SEQ_LEN) bool tensor. True = real position.
        t: (B,) float32 tensor of timesteps in [0, 1].
        rate_network: Module that maps (t, pad_mask) -> (B, SEQ_LEN) alpha.
        denoiser: BDDenoiser with .node_embedding and .edge_embedding attrs.
        vocab_config: VocabConfig for n_max (node/edge split point).
        gumbel_temperature: STGS temperature. Default 1.0.

    Returns:
        STGSOutput dict with soft_embeddings, x_t, mask_indicators,
        alpha_per_pos, and gumbel_weights.
    """
    B, SEQ_LEN = x0.shape
    n_max = vocab_config.n_max
    n_edges = SEQ_LEN - n_max
    device = x0.device

    # 1. Per-position alpha from rate network
    alpha = rate_network(t, pad_mask)  # (B, SEQ_LEN)

    # 2. STGS sampling
    gumbel_weights = stgs_sample(alpha, gumbel_temperature)  # (B, SEQ_LEN, 2)

    # 3. PAD override: force keep=1, mask=0 for PAD positions
    pad_positions = ~pad_mask  # (B, SEQ_LEN)
    gumbel_weights[pad_positions] = torch.tensor([1.0, 0.0], device=device)

    # 4. Extract keep/mask weights for soft embedding interpolation
    w_keep = gumbel_weights[:, :, 0].unsqueeze(-1)  # (B, SEQ_LEN, 1)
    w_mask = gumbel_weights[:, :, 1].unsqueeze(-1)  # (B, SEQ_LEN, 1)

    # 5. Build soft embeddings by mixing clean and mask token embeddings
    # --- Nodes (positions 0..n_max-1) ---
    clean_node_emb = denoiser.node_embedding(x0[:, :n_max])  # (B, n_max, d_model)
    mask_idx_node = torch.full(
        (B, n_max), NODE_MASK_IDX, dtype=torch.long, device=device
    )
    mask_node_emb = denoiser.node_embedding(mask_idx_node)  # (B, n_max, d_model)
    soft_node = (
        w_keep[:, :n_max] * clean_node_emb + w_mask[:, :n_max] * mask_node_emb
    )

    # --- Edges (positions n_max..seq_len-1) ---
    clean_edge_emb = denoiser.edge_embedding(x0[:, n_max:])  # (B, n_edges, d_model)
    mask_idx_edge = torch.full(
        (B, n_edges), EDGE_MASK_IDX, dtype=torch.long, device=device
    )
    mask_edge_emb = denoiser.edge_embedding(mask_idx_edge)  # (B, n_edges, d_model)
    soft_edge = (
        w_keep[:, n_max:] * clean_edge_emb + w_mask[:, n_max:] * mask_edge_emb
    )

    # Concatenate into full sequence
    soft_emb = torch.cat([soft_node, soft_edge], dim=1)  # (B, SEQ_LEN, d_model)

    # 6. Get discrete x_t from hard decisions
    hard_mask = gumbel_weights[:, :, 1] > 0.5  # (B, SEQ_LEN) bool

    mask_tokens = torch.full(
        (SEQ_LEN,), NODE_MASK_IDX, dtype=torch.long, device=device
    )
    mask_tokens[n_max:] = EDGE_MASK_IDX
    mask_tokens = mask_tokens.unsqueeze(0).expand(B, -1)  # (B, SEQ_LEN)

    x_t = torch.where(hard_mask, mask_tokens, x0)

    # 7. mask_indicators: True only where actively masked AND non-PAD
    mask_indicators = hard_mask & pad_mask

    return STGSOutput(
        soft_embeddings=soft_emb,
        x_t=x_t,
        mask_indicators=mask_indicators,
        alpha_per_pos=alpha,
        gumbel_weights=gumbel_weights,
    )


def forward_mask_eval_learned(
    x0: Tensor,
    pad_mask: Tensor,
    t: Tensor,
    rate_network: torch.nn.Module,
    vocab_config: VocabConfig = RPLAN_VOCAB_CONFIG,
) -> tuple[Tensor, Tensor]:
    """Forward masking for eval with learned rates (discrete, no STGS).

    At evaluation time we do not need differentiable soft embeddings.
    This function uses the rate network's alpha to make independent Bernoulli
    masking decisions, identical to v1 forward_mask but with per-position
    alpha from the learned rate network instead of a global noise schedule.

    Same return signature as v1 forward_mask: (x_t, mask_indicators).

    Args:
        x0: (B, SEQ_LEN) long tensor of clean token indices.
        pad_mask: (B, SEQ_LEN) bool tensor. True = real position.
        t: (B,) float32 tensor of timesteps in [0, 1].
        rate_network: Module that maps (t, pad_mask) -> (B, SEQ_LEN) alpha.
        vocab_config: VocabConfig for n_max. Defaults to RPLAN_VOCAB_CONFIG.

    Returns:
        Tuple of:
            x_t: (B, SEQ_LEN) long tensor with masked tokens.
            mask_indicators: (B, SEQ_LEN) bool tensor. True where masked.
    """
    B, SEQ_LEN = x0.shape
    n_max = vocab_config.n_max
    device = x0.device

    alpha = rate_network(t, pad_mask)  # (B, SEQ_LEN)
    rand = torch.rand(B, SEQ_LEN, device=device)
    should_mask = (rand >= alpha) & pad_mask

    mask_tokens = torch.full(
        (SEQ_LEN,), NODE_MASK_IDX, dtype=torch.long, device=device
    )
    mask_tokens[n_max:] = EDGE_MASK_IDX
    mask_tokens = mask_tokens.unsqueeze(0).expand(B, -1)

    x_t = torch.where(should_mask, mask_tokens, x0)
    return x_t, should_mask
