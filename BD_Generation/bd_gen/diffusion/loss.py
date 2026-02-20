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
        # Float64 for precision near t→0 where 1-alpha(t) is tiny.
        # See Zheng et al. (arXiv:2409.02908).
        alpha_t = noise_schedule.alpha(t_clamped.double())          # float64
        alpha_prime_t = noise_schedule.alpha_prime(t_clamped).double()

        denominator = 1.0 - alpha_t + self.eps
        w = -alpha_prime_t / denominator

        return torch.clamp(w, max=1000.0).float()

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

        # Clamp targets at non-loss positions to class 0 so CE never indexes
        # into -inf logits (MASK/PAD indices are clamped to -inf by the
        # denoiser's zero masking probabilities). The loss_mask zeros out
        # these positions anyway, so the target value is irrelevant.
        safe_node_x0 = torch.where(node_loss_mask, node_x0, torch.zeros_like(node_x0))
        safe_edge_x0 = torch.where(edge_loss_mask, edge_x0, torch.zeros_like(edge_x0))

        node_ce = F.cross_entropy(
            node_logits.reshape(-1, NODE_VOCAB_SIZE),
            safe_node_x0.reshape(-1),
            weight=self.node_class_weights,
            reduction="none",
        ).reshape(B, -1)  # (B, n_max)

        edge_ce = F.cross_entropy(
            edge_logits.reshape(-1, EDGE_VOCAB_SIZE),
            safe_edge_x0.reshape(-1),
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


class ELBOLossV2(nn.Module):
    """ELBO loss with per-position weights for learned forward process.

    Unlike v1 ELBOLoss which uses a scalar w(t) from a fixed noise schedule,
    this version receives per-position alpha and alpha_prime values from a
    learned rate network, computing per-position ELBO weights w_l(t).

    Additional differences from v1:
    - Separate normalization for node and edge losses (N_active_nodes and
      N_active_edges independently).
    - Lambda edge weighting for balancing node vs edge loss.
    - Does NOT take a noise_schedule — rates come from rate_network outputs.

    Args:
        edge_class_weights: (EDGE_VOCAB_SIZE,) inverse-frequency weights.
        node_class_weights: (NODE_VOCAB_SIZE,) or None.
        vocab_config: VocabConfig for n_max.
        lambda_edge: Relative weight for edge loss. Default 1.0.
        eps: Numerical stability epsilon. Default 1e-8.
        t_min: Minimum t for clamping. Default 1e-5.
        w_max: Maximum per-position weight clamp. Default 1000.0.
    """

    def __init__(
        self,
        edge_class_weights: Tensor,
        node_class_weights: Tensor | None = None,
        vocab_config: VocabConfig = RPLAN_VOCAB_CONFIG,
        lambda_edge: float = 1.0,
        eps: float = 1e-8,
        t_min: float = 1e-5,
        w_max: float = 1000.0,
    ) -> None:
        super().__init__()
        self.vocab_config = vocab_config
        self.lambda_edge = lambda_edge
        self.eps = eps
        self.t_min = t_min
        self.w_max = w_max

        self.register_buffer("edge_class_weights", edge_class_weights.clone())
        if node_class_weights is not None:
            self.register_buffer("node_class_weights", node_class_weights.clone())
        else:
            self.node_class_weights = None

    def forward(
        self,
        node_logits: Tensor,
        edge_logits: Tensor,
        x0: Tensor,
        pad_mask: Tensor,
        mask_indicators: Tensor,
        alpha_per_pos: Tensor,
        alpha_prime_per_pos: Tensor,
    ) -> Tensor:
        """Compute per-position ELBO loss.

        Args:
            node_logits: (B, n_max, NODE_VOCAB_SIZE) float32.
            edge_logits: (B, n_edges, EDGE_VOCAB_SIZE) float32.
            x0: (B, SEQ_LEN) long — original clean tokens.
            pad_mask: (B, SEQ_LEN) bool — True = real position.
            mask_indicators: (B, SEQ_LEN) bool — True = was masked.
            alpha_per_pos: (B, SEQ_LEN) from rate network.
            alpha_prime_per_pos: (B, SEQ_LEN) from rate network.

        Returns:
            Scalar float32 loss, averaged across the batch.
        """
        n_max = self.vocab_config.n_max

        # --- Split into node and edge parts ---
        node_x0 = x0[:, :n_max]
        edge_x0 = x0[:, n_max:]

        node_pad = pad_mask[:, :n_max]
        edge_pad = pad_mask[:, n_max:]

        node_mask_ind = mask_indicators[:, :n_max]
        edge_mask_ind = mask_indicators[:, n_max:]

        # --- Loss mask: masked AND real (not PAD) ---
        node_loss_mask = node_mask_ind & node_pad
        edge_loss_mask = edge_mask_ind & edge_pad

        # --- Per-position ELBO weight (float64 for precision) ---
        alpha_64 = alpha_per_pos.double()
        alpha_prime_64 = alpha_prime_per_pos.double()
        denominator = 1.0 - alpha_64 + self.eps
        w_per_pos = (-alpha_prime_64 / denominator).float()  # (B, SEQ_LEN)
        w_per_pos = torch.clamp(w_per_pos, max=self.w_max)

        # Split weights
        w_node = w_per_pos[:, :n_max]   # (B, n_max)
        w_edge = w_per_pos[:, n_max:]   # (B, n_edges)

        B = node_logits.shape[0]

        # --- Safe CE targets (from v1) ---
        safe_node_x0 = torch.where(node_loss_mask, node_x0, torch.zeros_like(node_x0))
        safe_edge_x0 = torch.where(edge_loss_mask, edge_x0, torch.zeros_like(edge_x0))

        # --- Per-position CE (unreduced) ---
        node_ce = F.cross_entropy(
            node_logits.reshape(-1, NODE_VOCAB_SIZE),
            safe_node_x0.reshape(-1),
            weight=self.node_class_weights,
            reduction="none",
        ).reshape(B, -1)  # (B, n_max)

        edge_ce = F.cross_entropy(
            edge_logits.reshape(-1, EDGE_VOCAB_SIZE),
            safe_edge_x0.reshape(-1),
            weight=self.edge_class_weights,
            reduction="none",
        ).reshape(B, -1)  # (B, n_edges)

        # --- Apply loss mask AND per-position weight ---
        node_ce = node_ce * node_loss_mask.float() * w_node
        edge_ce = edge_ce * edge_loss_mask.float() * w_edge

        # --- Separate normalization ---
        n_active_nodes = node_loss_mask.sum(dim=1).float().clamp(min=1.0)  # (B,)
        n_active_edges = edge_loss_mask.sum(dim=1).float().clamp(min=1.0)  # (B,)

        node_loss = node_ce.sum(dim=1) / n_active_nodes   # (B,)
        edge_loss = edge_ce.sum(dim=1) / n_active_edges   # (B,)

        # --- Combine with lambda_edge ---
        per_sample_loss = node_loss + self.lambda_edge * edge_loss  # (B,)

        return per_sample_loss.mean()
