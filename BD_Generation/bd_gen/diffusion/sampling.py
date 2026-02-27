"""Reverse sampling (denoising) for MDLM masked diffusion.

Iterates from t=1 (fully masked) to t=0 (clean), progressively unmasking
tokens. At each step, the model predicts logits for masked positions,
and tokens are stochastically or deterministically unmasked based on the
schedule's unmasking probability.

Supports fixed tokens for inpainting and an optional remasking function
for ReMDM-style inference.
"""

from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from bd_gen.data.vocab import (
    EDGE_MASK_IDX,
    EDGE_PAD_IDX,
    NODE_MASK_IDX,
    NODE_PAD_IDX,
    VocabConfig,
)
from bd_gen.diffusion.noise_schedule import NoiseSchedule

logger = logging.getLogger(__name__)


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


def _top_p_sample(logits: Tensor, top_p: float) -> Tensor:
    """Sample from the top-p (nucleus) of the probability distribution.

    Keeps the smallest set of tokens whose cumulative probability >= top_p,
    zeros out the rest, and samples from the filtered distribution.
    Uses temperature=1.0 (raw softmax). See Holtzman et al. (2020).

    Args:
        logits: (..., vocab_size) float32 logits.
        top_p: Cumulative probability threshold in (0, 1].

    Returns:
        (...,) long tensor of sampled token indices.
    """
    orig_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)  # (N, V)

    probs = torch.softmax(flat_logits, dim=-1)
    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
    cumulative_probs = sorted_probs.cumsum(dim=-1)

    # Mask tokens beyond the nucleus (keep at least top-1 token)
    sorted_mask = (cumulative_probs - sorted_probs) >= top_p
    sorted_probs[sorted_mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    # Sample from filtered distribution
    sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)  # (N, 1)

    # Map back to original vocabulary indices
    sampled_idx = sorted_indices.gather(-1, sampled_sorted_idx).squeeze(-1)  # (N,)
    return sampled_idx.reshape(orig_shape)


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


def _single_step_unmask(
    x_t: Tensor,
    node_logits: Tensor,
    edge_logits: Tensor,
    pad_mask: Tensor,
    p_unmask: Tensor,
    i: int,
    num_steps: int,
    n_max: int,
    top_p: float | None,
    temperature: float,
    unmasking_mode: str,
    device: torch.device,
    fixed_tokens: Tensor | None,
    fixed_mask: Tensor | None,
) -> Tensor:
    """Perform token selection, unmasking, PAD clamping, and inpainting.

    Corresponds to steps 4c–4h of the sampling loop.  The model call is
    NOT included — logits are passed in.  No remasking is applied.

    Args:
        x_t: (B, SEQ_LEN) current tokens.
        node_logits: (B, n_max, NODE_VOCAB_SIZE) model predictions.
        edge_logits: (B, n_edges, EDGE_VOCAB_SIZE) model predictions.
        pad_mask: (B, SEQ_LEN) bool, True=real.
        p_unmask: (B, 1) or (B, SEQ_LEN) unmasking probability.
        i: Current step index (counting down from num_steps-1 to 0).
        num_steps: Total number of denoising steps N.
        n_max: Number of node positions.
        top_p: Nucleus sampling threshold, or None.
        temperature: Sampling temperature (ignored when top_p is set).
        unmasking_mode: "random" or "llada".
        device: Torch device.
        fixed_tokens: Optional (B, SEQ_LEN) inpainting tokens.
        fixed_mask: Optional (B, SEQ_LEN) bool inpainting mask.

    Returns:
        (B, SEQ_LEN) updated token tensor.
    """
    batch_size = x_t.shape[0]
    seq_len = x_t.shape[1]

    # 4c. Choose predicted tokens
    if top_p is not None and top_p < 1.0:
        node_pred = _top_p_sample(node_logits, top_p)
        edge_pred = _top_p_sample(edge_logits, top_p)
    elif temperature == 0.0:
        node_pred = node_logits.argmax(dim=-1)
        edge_pred = edge_logits.argmax(dim=-1)
    else:
        node_gumbel = _gumbel_sample(node_logits, temperature, device)
        edge_gumbel = _gumbel_sample(edge_logits, temperature, device)
        node_pred = node_gumbel.argmax(dim=-1)
        edge_pred = edge_gumbel.argmax(dim=-1)

    pred_tokens = torch.cat([node_pred, edge_pred], dim=1)

    # 4d. Identify currently masked positions
    is_node_mask = x_t[:, :n_max] == NODE_MASK_IDX
    is_edge_mask = x_t[:, n_max:] == EDGE_MASK_IDX
    is_mask = torch.cat([is_node_mask, is_edge_mask], dim=1)

    # 4e. Decide which MASK positions to unmask
    if unmasking_mode == "random":
        unmask_rand = torch.rand(batch_size, seq_len, device=device)
        should_unmask = (unmask_rand < p_unmask) & is_mask
    elif unmasking_mode == "llada":
        node_probs = F.softmax(node_logits, dim=-1)
        edge_probs = F.softmax(edge_logits, dim=-1)
        node_conf = node_probs.gather(
            -1, node_pred.unsqueeze(-1)
        ).squeeze(-1)
        edge_conf = edge_probs.gather(
            -1, edge_pred.unsqueeze(-1)
        ).squeeze(-1)
        confidence = torch.cat([node_conf, edge_conf], dim=1)

        confidence = torch.where(
            is_mask, confidence,
            torch.tensor(-float("inf"), device=device),
        )

        num_masked = is_mask.sum(dim=1)
        if p_unmask.shape[-1] > 1:
            # Per-position (v2): sum p_unmask over masked positions
            budget = (p_unmask * is_mask.float()).sum(dim=1)
            num_to_unmask = budget.round().long().clamp(min=1)
        else:
            num_to_unmask = (
                (p_unmask.squeeze(1) * num_masked.float())
                .round().long().clamp(min=1)
            )

        if i == 0:
            should_unmask = is_mask
        else:
            should_unmask = torch.zeros_like(is_mask)
            for b in range(batch_size):
                k = int(min(num_to_unmask[b].item(), num_masked[b].item()))
                if k > 0:
                    _, topk_idx = torch.topk(confidence[b], k)
                    should_unmask[b, topk_idx] = True
    else:
        raise ValueError(
            f"Unknown unmasking_mode: {unmasking_mode!r}. "
            "Use 'random' or 'llada'."
        )

    # 4f. Update x_t
    x_t = torch.where(should_unmask, pred_tokens, x_t)

    # 4g. Clamp PAD positions
    x_t = _clamp_pad(x_t, pad_mask, n_max)

    # 4h. Apply fixed_tokens for inpainting
    if fixed_tokens is not None and fixed_mask is not None:
        x_t = torch.where(fixed_mask, fixed_tokens.to(device), x_t)

    return x_t


def _single_step_remask(
    x_t: Tensor,
    remasking_fn: Callable | None,
    t_now: float,
    t_next: float,
    t_switch: float,
    i: int,
    pad_mask: Tensor,
    node_logits: Tensor,
    edge_logits: Tensor,
) -> Tensor:
    """Apply remasking if applicable (step 4i).

    No-op if ``remasking_fn`` is None, or if the step/switch conditions
    are not met (i == 0 or t_now >= t_switch).

    Args:
        x_t: (B, SEQ_LEN) current tokens.
        remasking_fn: Optional callable for ReMDM remasking.
        t_now: Current timestep.
        t_next: Next timestep.
        t_switch: Timestep threshold for activating remasking.
        i: Current step index.
        pad_mask: (B, SEQ_LEN) bool.
        node_logits: (B, n_max, NODE_VOCAB_SIZE) model predictions.
        edge_logits: (B, n_edges, EDGE_VOCAB_SIZE) model predictions.

    Returns:
        (B, SEQ_LEN) possibly remasked token tensor.
    """
    if remasking_fn is not None and i > 0 and t_now < t_switch:
        x_t = remasking_fn(
            x_t, t_now, t_next, pad_mask,
            node_logits=node_logits, edge_logits=edge_logits,
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
    top_p: float | None = None,
    unmasking_mode: str = "random",
    t_switch: float = 1.0,
    fixed_tokens: Tensor | None = None,
    fixed_mask: Tensor | None = None,
    remasking_fn: Callable | None = None,
    rate_network: torch.nn.Module | None = None,
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
            Ignored when top_p is set.
        top_p: If not None and < 1.0, use nucleus (top-p) sampling.
            Keeps smallest set of tokens whose cumulative probability
            >= top_p, then samples from the filtered distribution at
            temperature=1.0. Takes priority over temperature.
        unmasking_mode: Strategy for selecting which MASK positions to
            unmask at each step. "random" = probabilistic coin-flip per
            position (original MDLM). "llada" = unmask highest-
            confidence positions first (LLaDA-style, Nie et al.).
        t_switch: Timestep threshold for activating remasking. Remasking
            is only applied when t_now < t_switch. Default 1.0 means
            remasking at all steps (backward compatible). Lower values
            delay remasking until later in denoising (ReMDM-Switch).
        fixed_tokens: (B, SEQ_LEN) long tensor of tokens to keep fixed
            (for inpainting). Used together with fixed_mask.
        fixed_mask: (B, SEQ_LEN) bool tensor. True = position is fixed
            (clamp to fixed_tokens after each step).
        remasking_fn: Optional callable(x_t, t_now, t_next, pad_mask,
            node_logits=..., edge_logits=...) -> x_t_remasked.
            For ReMDM inference. Applied after each unmasking step.
            Not called at the final step (i=0) to ensure all tokens
            are decoded. Receives model logits for confidence-based
            remasking strategies.
        rate_network: Optional rate network (v2). If provided, uses
            per-position alpha_l(t) instead of scalar alpha(t) from
            noise_schedule. When None (default), uses v1 behavior.
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

        # 4b. Compute unmasking probability
        if rate_network is not None:
            # v2 path: per-position alpha from rate_network
            alpha_now_64 = rate_network(t_tensor, pad_mask).double()  # (B, SEQ_LEN)
            t_next_tensor = torch.full(
                (batch_size,), t_next, dtype=torch.float32, device=device,
            )
            alpha_next_64 = rate_network(  # (B, SEQ_LEN)
                t_next_tensor, pad_mask,
            ).double()
            p_unmask = (alpha_next_64 - alpha_now_64) / (1.0 - alpha_now_64 + 1e-8)
            p_unmask = torch.clamp(p_unmask, min=0.0, max=1.0).float()
            # p_unmask is (B, SEQ_LEN) — per-position
        else:
            # v1 path: scalar alpha from noise_schedule
            # Float64 to prevent catastrophic cancellation when alpha
            # values are close (high num_steps).
            # See Zheng et al. (arXiv:2409.02908).
            alpha_now_64 = noise_schedule.alpha(t_tensor.double())  # (B,) float64
            alpha_next_64 = noise_schedule.alpha(
                torch.full((batch_size,), t_next, dtype=torch.float64, device=device)
            )  # (B,) float64
            p_unmask = (alpha_next_64 - alpha_now_64) / (1.0 - alpha_now_64 + 1e-8)
            p_unmask = torch.clamp(p_unmask, min=0.0, max=1.0).float()
            p_unmask = p_unmask.unsqueeze(1)  # (B, 1) for broadcasting

        # 4c-4h. Token selection + unmasking + PAD clamp + inpainting
        x_t = _single_step_unmask(
            x_t, node_logits, edge_logits, pad_mask, p_unmask,
            i, num_steps, n_max, top_p, temperature, unmasking_mode,
            torch.device(device), fixed_tokens, fixed_mask,
        )

        # 4i. Apply remasking if provided (ReMDM)
        x_t = _single_step_remask(
            x_t, remasking_fn, t_now, t_next, t_switch, i,
            pad_mask, node_logits, edge_logits,
        )

    # --- Step 5: Final cleanup (safety net) ---
    # With SUBS zero masking probabilities enforced in the denoiser (MASK/PAD
    # logits clamped to -inf), this step should be a no-op in practice.
    # Kept as defense-in-depth for edge cases (e.g., remasking_fn
    # introducing MASK tokens after the last denoising step).
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
