"""MDLM continuous-time ELBO loss with PAD exclusion and class weighting.

Implements the loss from MDLM (Sahoo et al., arXiv:2406.07524) adapted
for our dual-vocabulary (node + edge) token sequences. The loss is:

    L = E_t [ w(t) * (1/N_active) * sum_l CE(logits_l, x0_l) * loss_mask_l ]

where w(t) = -alpha'(t) / (1 - alpha(t) + eps) is the per-timestep ELBO
weight, and N_active is the per-sample count of masked non-PAD positions.

Key properties:
- PAD positions contribute exactly zero to the loss (enforced by loss_mask).
- Per-sample normalization by N_active ensures small and large graphs
  contribute equally to the gradient.
- Edge positions use class-weighted CE; node CE is optionally weighted.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bd_gen.data.vocab import (
    EDGE_VOCAB_SIZE,
    NODE_VOCAB_SIZE,
    RPLAN_VOCAB_CONFIG,
    VocabConfig,
)
from bd_gen.diffusion.noise_schedule import NoiseSchedule


class ELBOLoss(nn.Module):
    """MDLM continuous-time ELBO loss.

    Splits logits, targets, and masks into node and edge parts, computes
    separate cross-entropy for each vocabulary, applies ELBO timestep
    weighting, and normalizes per-sample by the number of loss-active
    positions.

    Args:
        edge_class_weights: (EDGE_VOCAB_SIZE,) float32 tensor of inverse-
            frequency weights for edge classes.
        node_class_weights: (NODE_VOCAB_SIZE,) float32 tensor. None means
            unweighted CE for nodes (v1 default).
        vocab_config: VocabConfig for n_max. Defaults to RPLAN_VOCAB_CONFIG.
        eps: Epsilon for numerical stability in w(t) computation.
        t_min: Minimum timestep for clamping to prevent w(t) -> inf.
    """

    def __init__(
        self,
        edge_class_weights: Tensor,
        node_class_weights: Tensor | None = None,
        vocab_config: VocabConfig = RPLAN_VOCAB_CONFIG,
        eps: float = 1e-8,
        t_min: float = 1e-5,
    ) -> None:
        super().__init__()
        self.vocab_config = vocab_config
        self.eps = eps
        self.t_min = t_min

        self.register_buffer("edge_class_weights", edge_class_weights.clone())
        if node_class_weights is not None:
            self.register_buffer("node_class_weights", node_class_weights.clone())
        else:
            self.node_class_weights = None

    def _compute_w(self, t: Tensor, noise_schedule: NoiseSchedule) -> Tensor:
        """Compute ELBO weight: w(t) = -alpha'(t) / (1 - alpha(t) + eps).

        Clamps t to [t_min, 1.0] and w to [0, 1000] for stability.

        Args:
            t: (B,) float32 timesteps.
            noise_schedule: Schedule providing alpha and alpha_prime.

        Returns:
            (B,) float32 ELBO weights.
        """
        t_clamped = torch.clamp(t, min=self.t_min, max=1.0)
        alpha_t = noise_schedule.alpha(t_clamped)
        alpha_prime_t = noise_schedule.alpha_prime(t_clamped)

        denominator = 1.0 - alpha_t + self.eps
        w = -alpha_prime_t / denominator

        return torch.clamp(w, max=1000.0)

    def forward(
        self,
        node_logits: Tensor,
        edge_logits: Tensor,
        x0: Tensor,
        x_t: Tensor,
        pad_mask: Tensor,
        mask_indicators: Tensor,
        t: Tensor,
        noise_schedule: NoiseSchedule,
    ) -> Tensor:
        """Compute the ELBO loss.

        Args:
            node_logits: (B, n_max, NODE_VOCAB_SIZE) float32.
            edge_logits: (B, n_edges, EDGE_VOCAB_SIZE) float32.
            x0: (B, SEQ_LEN) long — original clean tokens.
            x_t: (B, SEQ_LEN) long — noised tokens (unused in v1 but
                available for future extensions).
            pad_mask: (B, SEQ_LEN) bool — True = real position.
            mask_indicators: (B, SEQ_LEN) bool — True = was masked.
            t: (B,) float32 — timesteps.
            noise_schedule: Schedule for computing w(t).

        Returns:
            Scalar float32 loss, averaged across the batch.
        """
        n_max = self.vocab_config.n_max

        # --- Split into node and edge parts ---
        node_x0 = x0[:, :n_max]                    # (B, n_max)
        edge_x0 = x0[:, n_max:]                    # (B, n_edges)

        node_pad = pad_mask[:, :n_max]              # (B, n_max)
        edge_pad = pad_mask[:, n_max:]              # (B, n_edges)

        node_mask_ind = mask_indicators[:, :n_max]  # (B, n_max)
        edge_mask_ind = mask_indicators[:, n_max:]  # (B, n_edges)

        # --- Loss mask: masked AND real (not PAD) ---
        node_loss_mask = node_mask_ind & node_pad   # (B, n_max)
        edge_loss_mask = edge_mask_ind & edge_pad   # (B, n_edges)

        # --- Per-position CE (unreduced) ---
        # Reshape to (B*L, V) for F.cross_entropy, then back to (B, L)
        B = node_logits.shape[0]

        node_ce = F.cross_entropy(
            node_logits.reshape(-1, NODE_VOCAB_SIZE),
            node_x0.reshape(-1),
            weight=self.node_class_weights,
            reduction="none",
        ).reshape(B, -1)  # (B, n_max)

        edge_ce = F.cross_entropy(
            edge_logits.reshape(-1, EDGE_VOCAB_SIZE),
            edge_x0.reshape(-1),
            weight=self.edge_class_weights,
            reduction="none",
        ).reshape(B, -1)  # (B, n_edges)

        # --- Apply loss mask (zero out non-active positions) ---
        node_ce = node_ce * node_loss_mask.float()
        edge_ce = edge_ce * edge_loss_mask.float()

        # --- Per-sample sum ---
        per_sample_loss = node_ce.sum(dim=1) + edge_ce.sum(dim=1)  # (B,)

        # --- Per-sample normalization by N_active ---
        n_active = (
            node_loss_mask.sum(dim=1) + edge_loss_mask.sum(dim=1)
        ).float()  # (B,)
        n_active = torch.clamp(n_active, min=1.0)
        per_sample_loss = per_sample_loss / n_active  # (B,)

        # --- ELBO weight w(t) ---
        w = self._compute_w(t, noise_schedule)  # (B,)
        weighted_loss = w * per_sample_loss  # (B,)

        return weighted_loss.mean()
