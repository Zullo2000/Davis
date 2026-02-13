"""Reverse sampling (denoising) for MDLM masked diffusion.

Iterates from t=1 (fully masked) to t=0 (clean), progressively unmasking
tokens. At each step, the model predicts logits for masked positions,
and tokens are stochastically or deterministically unmasked based on the
schedule's unmasking probability.

Supports pluggable guidance functions, fixed tokens for inpainting, and
an optional remasking function for ReMDM-style inference.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from bd_gen.data.vocab import (
    EDGE_MASK_IDX,
    EDGE_PAD_IDX,
    NODE_MASK_IDX,
    NODE_PAD_IDX,
    VocabConfig,
)
from bd_gen.diffusion.noise_schedule import NoiseSchedule


def _gumbel_sample(logits: Tensor, temperature: float, device: torch.device) -> Tensor:
    """Add Gumbel noise to logits for stochastic sampling.

    Uses float64 for the Gumbel noise computation to prevent underflow
    in log(-log(u)) when u is very close to 0 or 1.

    Args:
        logits: (..., vocab_size) float32 logits.
        temperature: > 0 temperature scaling.
        device: Device for output.

    Returns:
        Perturbed logits of same shape as input, dtype float32.
    """
    u = torch.rand(logits.shape, dtype=torch.float64, device=device)
    u = torch.clamp(u, min=1e-10, max=1.0 - 1e-10)
    gumbel_noise = -torch.log(-torch.log(u))

    perturbed = logits.double() / temperature + gumbel_noise
    return perturbed.float()


def _clamp_pad(
    x_t: Tensor,
    pad_mask: Tensor,
    n_max: int,
) -> Tensor:
    """Clamp PAD positions to their correct PAD token values.

    Args:
        x_t: (B, SEQ_LEN) long tensor.
        pad_mask: (B, SEQ_LEN) bool, True=real.
        n_max: Number of node positions at the start of the sequence.

    Returns:
        x_t with PAD positions clamped (modified in-place and returned).
    """
    device = x_t.device
    x_t[:, :n_max] = torch.where(
        pad_mask[:, :n_max],
        x_t[:, :n_max],
        torch.tensor(NODE_PAD_IDX, dtype=torch.long, device=device),
    )
    x_t[:, n_max:] = torch.where(
        pad_mask[:, n_max:],
        x_t[:, n_max:],
        torch.tensor(EDGE_PAD_IDX, dtype=torch.long, device=device),
    )
    return x_t


@torch.no_grad()
def sample(
    model: torch.nn.Module,
    noise_schedule: NoiseSchedule,
    vocab_config: VocabConfig,
    batch_size: int,
    num_steps: int,
    temperature: float = 0.0,
    guidance_fn: Callable | None = None,
    fixed_tokens: Tensor | None = None,
    fixed_mask: Tensor | None = None,
    remasking_fn: Callable | None = None,
    num_rooms_distribution: Tensor | None = None,
    fixed_num_rooms: int | None = None,
    device: str = "cpu",
) -> Tensor:
    """Generate bubble diagram token sequences via reverse diffusion.

    Args:
        model: BDDenoiser (or any model matching the forward signature).
            Called as model(tokens, pad_mask, t) -> (node_logits, edge_logits).
        noise_schedule: NoiseSchedule for alpha(t) computation.
        vocab_config: VocabConfig providing n_max, seq_len, compute_pad_mask().
        batch_size: Number of samples to generate.
        num_steps: Number of discrete denoising steps (N).
        temperature: Sampling temperature. 0.0 = deterministic argmax.
            > 0.0 = stochastic via Gumbel-perturbed logits.
        guidance_fn: Optional callable that receives
            ``((node_logits, edge_logits), x_t, t_scalar, pad_mask)``
            and returns a modified ``(node_logits, edge_logits)`` tuple.
        fixed_tokens: (B, SEQ_LEN) long tensor of tokens to keep fixed
            (for inpainting). Used together with fixed_mask.
        fixed_mask: (B, SEQ_LEN) bool tensor. True = position is fixed
            (clamp to fixed_tokens after each step).
        remasking_fn: Optional callable(x_t, t) -> x_t_remasked.
            For ReMDM inference. Applied after each unmasking step.
        num_rooms_distribution: (n_max,) float32 histogram. Index k =
            P(k+1 rooms). If None, uniform over [1, n_max].
        fixed_num_rooms: If given, all samples have this many rooms.
            Overrides num_rooms_distribution.
        device: Target device string.

    Returns:
        (B, SEQ_LEN) long tensor of generated token sequences.
        Non-PAD positions contain valid room/edge types (no MASK tokens).
        PAD positions contain NODE_PAD_IDX or EDGE_PAD_IDX.
    """
    n_max = vocab_config.n_max
    seq_len = vocab_config.seq_len

    # --- Step 1: Determine num_rooms per sample ---
    if fixed_num_rooms is not None:
        num_rooms_list = [fixed_num_rooms] * batch_size
    elif num_rooms_distribution is not None:
        room_indices = torch.multinomial(
            num_rooms_distribution, batch_size, replacement=True
        )
        num_rooms_list = (room_indices + 1).tolist()  # index k -> k+1 rooms
    else:
        num_rooms_list = torch.randint(1, n_max + 1, (batch_size,)).tolist()

    # --- Step 2: Compute pad_mask ---
    pad_mask = torch.stack(
        [vocab_config.compute_pad_mask(nr) for nr in num_rooms_list]
    ).to(device)  # (B, SEQ_LEN)

    # --- Step 3: Initialize x_t (fully masked) ---
    x_t = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    # Node positions: MASK for real, PAD for padding
    for b in range(batch_size):
        nr = num_rooms_list[b]
        x_t[b, :nr] = NODE_MASK_IDX
        x_t[b, nr:n_max] = NODE_PAD_IDX

    # Edge positions: MASK where real, PAD where padding
    edge_pad_mask = pad_mask[:, n_max:]  # (B, n_edges)
    x_t[:, n_max:] = torch.where(
        edge_pad_mask,
        torch.tensor(EDGE_MASK_IDX, dtype=torch.long, device=device),
        torch.tensor(EDGE_PAD_IDX, dtype=torch.long, device=device),
    )

    # --- Step 4: Reverse loop ---
    for i in range(num_steps - 1, -1, -1):
        t_now = (i + 1) / num_steps
        t_next = i / num_steps

        t_tensor = torch.full(
            (batch_size,), t_now, dtype=torch.float32, device=device
        )

        # 4a. Get model predictions
        node_logits, edge_logits = model(x_t, pad_mask, t_tensor)

        # 4b. Apply guidance if provided
        if guidance_fn is not None:
            node_logits, edge_logits = guidance_fn(
                (node_logits, edge_logits), x_t, t_now, pad_mask
            )

        # 4c. Compute unmasking probability
        alpha_now = noise_schedule.alpha(t_tensor)  # (B,)
        alpha_next = noise_schedule.alpha(
            torch.full_like(t_tensor, t_next)
        )  # (B,)
        p_unmask = (alpha_next - alpha_now) / (1.0 - alpha_now + 1e-8)
        p_unmask = torch.clamp(p_unmask, min=0.0, max=1.0)
        p_unmask = p_unmask.unsqueeze(1)  # (B, 1) for broadcasting

        # 4d. Decide which MASK positions to unmask
        unmask_rand = torch.rand(batch_size, seq_len, device=device)
        should_unmask = unmask_rand < p_unmask  # (B, SEQ_LEN)

        # Identify currently masked positions
        is_node_mask = x_t[:, :n_max] == NODE_MASK_IDX  # (B, n_max)
        is_edge_mask = x_t[:, n_max:] == EDGE_MASK_IDX  # (B, n_edges)
        is_mask = torch.cat([is_node_mask, is_edge_mask], dim=1)  # (B, SEQ_LEN)

        # Only unmask positions that are currently MASK
        should_unmask = should_unmask & is_mask

        # 4e. Choose tokens for unmasked positions
        if temperature == 0.0:
            node_pred = node_logits.argmax(dim=-1)  # (B, n_max)
            edge_pred = edge_logits.argmax(dim=-1)  # (B, n_edges)
        else:
            node_gumbel = _gumbel_sample(node_logits, temperature, device)
            edge_gumbel = _gumbel_sample(edge_logits, temperature, device)
            node_pred = node_gumbel.argmax(dim=-1)
            edge_pred = edge_gumbel.argmax(dim=-1)

        pred_tokens = torch.cat([node_pred, edge_pred], dim=1)  # (B, SEQ_LEN)

        # 4f. Update x_t: unmask selected positions, keep rest
        x_t = torch.where(should_unmask, pred_tokens, x_t)

        # 4g. Clamp PAD positions
        x_t = _clamp_pad(x_t, pad_mask, n_max)

        # 4h. Apply fixed_tokens for inpainting
        if fixed_tokens is not None and fixed_mask is not None:
            x_t = torch.where(fixed_mask, fixed_tokens.to(device), x_t)

        # 4i. Apply remasking if provided (ReMDM)
        if remasking_fn is not None:
            x_t = remasking_fn(x_t, t_next)

    # --- Step 5: Final cleanup ---
    # Any remaining MASK tokens -> predict at t ~ 0
    remaining_node_mask = x_t[:, :n_max] == NODE_MASK_IDX
    remaining_edge_mask = x_t[:, n_max:] == EDGE_MASK_IDX
    has_remaining = remaining_node_mask.any() or remaining_edge_mask.any()

    if has_remaining:
        t_final = torch.full(
            (batch_size,), 1e-5, dtype=torch.float32, device=device
        )
        node_logits, edge_logits = model(x_t, pad_mask, t_final)
        node_pred = node_logits.argmax(dim=-1)
        edge_pred = edge_logits.argmax(dim=-1)

        x_t[:, :n_max] = torch.where(
            remaining_node_mask, node_pred, x_t[:, :n_max]
        )
        x_t[:, n_max:] = torch.where(
            remaining_edge_mask, edge_pred, x_t[:, n_max:]
        )

    # Final PAD enforcement
    x_t = _clamp_pad(x_t, pad_mask, n_max)

    return x_t
