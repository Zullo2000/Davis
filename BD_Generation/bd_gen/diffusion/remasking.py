"""ReMDM-style remasking for inference-time error correction.

Implements post-hoc remasking strategies from Schiff et al.,
"Remasking Discrete Diffusion Models with Inference-Time Scaling"
(arXiv:2503.00307). At each denoising step, already-decoded tokens
can be stochastically re-masked and re-predicted in subsequent steps,
enabling the model to correct early mistakes when more context is
available.

This is an inference-only enhancement — no retraining needed. The model
is trained with standard MDLM (sigma=0), and remasking is applied only
during sampling.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from bd_gen.data.vocab import (
    EDGE_MASK_IDX,
    NODE_MASK_IDX,
    VocabConfig,
)
from bd_gen.diffusion.noise_schedule import NoiseSchedule


class RemaskingSchedule:
    """Callable that stochastically re-masks decoded positions at each step.

    Strategies:
        - "cap": sigma_t = min(eta, sigma_max), where
          sigma_max = min(1, (1 - alpha_s) / alpha_t).
          Uniform probability for all decoded positions.
        - "rescale": sigma_t = eta * sigma_max.
          Uniform probability for all decoded positions.
        - "confidence": Per-position remasking probability based on model
          confidence (ReMDM paper Section 4.1). Low-confidence decoded
          tokens get higher remasking probability, high-confidence tokens
          get lower probability. The total remasking budget is sigma_max
          (determined by the noise schedule), distributed across positions
          by softmax(-confidence). The eta parameter is NOT used for this
          strategy — sigma_max acts as the natural budget controller.

    Args:
        strategy: Remasking strategy ("cap", "rescale", or "confidence").
        eta: Remasking intensity. 0 = no remasking, higher = more aggressive.
        noise_schedule: NoiseSchedule for alpha(t) computation.
        vocab_config: VocabConfig for n_max (node/edge position split).
    """

    def __init__(
        self,
        strategy: str,
        eta: float,
        noise_schedule: NoiseSchedule | None,  # None when using rate_network
        vocab_config: VocabConfig,
        rate_network: torch.nn.Module | None = None,  # NEW
    ) -> None:
        if strategy not in ("cap", "rescale", "confidence"):
            raise ValueError(
                f"Unknown remasking strategy: {strategy!r}. "
                "Use 'cap', 'rescale', or 'confidence'."
            )
        if eta < 0:
            raise ValueError(f"eta must be non-negative, got {eta}")
        if noise_schedule is None and rate_network is None:
            raise ValueError(
                "Either noise_schedule or rate_network must be provided."
            )
        self.strategy = strategy
        self.eta = eta
        self.noise_schedule = noise_schedule
        self.vocab_config = vocab_config
        self.rate_network = rate_network

    def _compute_sigma_max(
        self,
        t_now: float,
        t_next: float,
        batch_size: int,
        device: torch.device,
        pad_mask: Tensor | None = None,  # NEW: required for rate_network
    ) -> Tensor:
        """Compute the upper-bound remasking probability sigma_max in float64.

        sigma_max = clamp(min(1, (1 - alpha_s) / alpha_t), 0, 1)

        Args:
            t_now: Current timestep (noisier, higher t).
            t_next: Destination timestep (cleaner, lower t).
            batch_size: Batch size (for broadcasting).
            device: Target device.
            pad_mask: (B, SEQ_LEN) bool tensor, required for rate_network path.

        Returns:
            (B, 1) for v1 or (B, SEQ_LEN) for v2 float32 tensor.
        """
        if self.rate_network is not None:
            # v2 path: per-position alpha from rate network
            t_now_tensor = torch.full(
                (batch_size,), t_now, dtype=torch.float32, device=device,
            )
            t_next_tensor = torch.full(
                (batch_size,), t_next, dtype=torch.float32, device=device,
            )
            alpha_t = self.rate_network(t_now_tensor, pad_mask).double()
            alpha_s = self.rate_network(t_next_tensor, pad_mask).double()
            sigma_max = (1.0 - alpha_s) / (alpha_t + 1e-8)
            sigma_max = torch.clamp(sigma_max, min=0.0, max=1.0)
            return sigma_max.float()  # (B, SEQ_LEN)

        t_now_tensor = torch.full(
            (batch_size,), t_now, dtype=torch.float64, device=device
        )
        t_next_tensor = torch.full(
            (batch_size,), t_next, dtype=torch.float64, device=device
        )

        alpha_t = self.noise_schedule.alpha(t_now_tensor)  # (B,) float64
        alpha_s = self.noise_schedule.alpha(t_next_tensor)  # (B,) float64

        sigma_max = (1.0 - alpha_s) / (alpha_t + 1e-8)
        sigma_max = torch.clamp(sigma_max, min=0.0, max=1.0)

        return sigma_max.float().unsqueeze(1)  # (B, 1)

    def _compute_sigma_t(
        self,
        t_now: float,
        t_next: float,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        """Compute the remasking probability sigma_t in float64.

        Args:
            t_now: Current timestep (noisier, higher t).
            t_next: Destination timestep (cleaner, lower t).
            batch_size: Batch size (for broadcasting).
            device: Target device.

        Returns:
            (B, 1) float32 tensor of sigma values, broadcastable over seq_len.
        """
        sigma_max = self._compute_sigma_max(t_now, t_next, batch_size, device)

        if self.strategy == "cap":
            sigma_t = torch.minimum(
                torch.tensor(self.eta, dtype=torch.float32, device=device),
                sigma_max,
            )
        else:  # rescale
            sigma_t = self.eta * sigma_max

        return sigma_t  # (B, 1)

    def __call__(
        self,
        x_t: Tensor,
        t_now: float,
        t_next: float,
        pad_mask: Tensor,
        node_logits: Tensor | None = None,
        edge_logits: Tensor | None = None,
    ) -> Tensor:
        """Apply stochastic remasking to already-decoded positions.

        Args:
            x_t: (B, SEQ_LEN) long tensor of current tokens.
            t_now: Current timestep (noisier).
            t_next: Destination timestep (cleaner).
            pad_mask: (B, SEQ_LEN) bool tensor, True=real position.
            node_logits: (B, n_max, NODE_VOCAB_SIZE) float tensor. Required
                for strategy="confidence", ignored otherwise.
            edge_logits: (B, n_edges, EDGE_VOCAB_SIZE) float tensor. Required
                for strategy="confidence", ignored otherwise.

        Returns:
            (B, SEQ_LEN) long tensor with some positions re-masked.
        """
        if self.strategy != "confidence" and self.eta == 0.0:
            return x_t

        batch_size, seq_len = x_t.shape
        n_max = self.vocab_config.n_max
        device = x_t.device

        # Identify positions that are NOT currently masked (candidates for remasking)
        is_node_mask = x_t[:, :n_max] == NODE_MASK_IDX
        is_edge_mask = x_t[:, n_max:] == EDGE_MASK_IDX
        is_mask = torch.cat([is_node_mask, is_edge_mask], dim=1)

        # Remask candidates: non-MASK AND non-PAD (never remask PAD)
        remask_candidates = (~is_mask) & pad_mask

        if self.strategy == "confidence":
            should_remask = self._confidence_remasking(
                x_t, t_now, t_next, remask_candidates, is_mask, pad_mask,
                node_logits, edge_logits,
            )
        else:
            # cap or rescale: uniform sigma for all decoded positions
            sigma_t = self._compute_sigma_t(t_now, t_next, batch_size, device)
            remask_rand = torch.rand(batch_size, seq_len, device=device)
            should_remask = (remask_rand < sigma_t) & remask_candidates

        # Apply correct MASK token per position type
        node_remask = should_remask[:, :n_max]
        edge_remask = should_remask[:, n_max:]

        x_t = x_t.clone()
        x_t[:, :n_max] = torch.where(
            node_remask,
            torch.tensor(NODE_MASK_IDX, dtype=torch.long, device=device),
            x_t[:, :n_max],
        )
        x_t[:, n_max:] = torch.where(
            edge_remask,
            torch.tensor(EDGE_MASK_IDX, dtype=torch.long, device=device),
            x_t[:, n_max:],
        )

        return x_t

    def _confidence_remasking(
        self,
        x_t: Tensor,
        t_now: float,
        t_next: float,
        remask_candidates: Tensor,
        is_mask: Tensor,
        pad_mask: Tensor,
        node_logits: Tensor | None,
        edge_logits: Tensor | None,
    ) -> Tensor:
        """Compute per-position remasking decisions using model confidence.

        Follows the ReMDM confidence-based remasking (Shi et al., Section
        4.1): the remasking probability is redistributed across decoded
        positions based on model confidence. Low-confidence positions are
        more likely to be remasked, high-confidence positions less so.

        The total remasking budget is sigma_max * n_decoded (determined
        entirely by the noise schedule). The eta parameter is NOT used.
        Formula: sigma_t[l] = softmax(-confidence)[l] * sigma_max
        (ReMDM paper Section 4.1).

        Args:
            x_t: (B, SEQ_LEN) current tokens.
            t_now, t_next: Timestep pair.
            remask_candidates: (B, SEQ_LEN) bool, True=decoded non-PAD position.
            is_mask: (B, SEQ_LEN) bool, True=currently masked.
            pad_mask: (B, SEQ_LEN) bool, True=real position.
            node_logits: (B, n_max, NODE_VOCAB_SIZE) model logits.
            edge_logits: (B, n_edges, EDGE_VOCAB_SIZE) model logits.

        Returns:
            (B, SEQ_LEN) bool tensor of positions to remask.
        """
        if node_logits is None or edge_logits is None:
            raise ValueError(
                "strategy='confidence' requires node_logits and edge_logits"
            )

        batch_size, seq_len = x_t.shape
        n_max = self.vocab_config.n_max
        device = x_t.device

        # --- Step 1: Base sigma (sigma_max from noise schedule, no eta) ---
        sigma_max = self._compute_sigma_max(t_now, t_next, batch_size, device, pad_mask)
        # (B, 1) or (B, SEQ_LEN) — this is the total remasking budget

        # --- Step 2: Per-position confidence ---
        # Confidence = P(decoded_token) from the model's softmax output.
        # Gather the probability of each position's current token.
        node_probs = F.softmax(node_logits, dim=-1)  # (B, n_max, NODE_VOCAB)
        edge_probs = F.softmax(edge_logits, dim=-1)  # (B, n_edges, EDGE_VOCAB)

        node_tokens = x_t[:, :n_max]  # (B, n_max)
        edge_tokens = x_t[:, n_max:]  # (B, n_edges)

        node_conf = node_probs.gather(
            -1, node_tokens.unsqueeze(-1)
        ).squeeze(-1)  # (B, n_max)
        edge_conf = edge_probs.gather(
            -1, edge_tokens.unsqueeze(-1)
        ).squeeze(-1)  # (B, n_edges)
        confidence = torch.cat([node_conf, edge_conf], dim=1)  # (B, SEQ_LEN)

        # --- Step 3: Remasking weights via softmax(-confidence) ---
        # Low confidence → high exp(-conf) → high remasking weight.
        # Masked and PAD positions get -inf → exp(-inf) = 0 in softmax.
        neg_conf = -confidence
        neg_conf = torch.where(
            remask_candidates,
            neg_conf,
            torch.tensor(float("-inf"), device=device),
        )
        # softmax over positions within each sample
        eta_conf = F.softmax(neg_conf, dim=-1)  # (B, SEQ_LEN), sums to ~1

        # --- Step 4: Scale to sigma_max budget ---
        # eta_conf sums to 1, so average over decoded positions = 1/n_decoded.
        # Multiply by n_decoded so the average per-position sigma = sigma_max.
        n_decoded = remask_candidates.sum(dim=1, keepdim=True).float()
        n_decoded = n_decoded.clamp(min=1.0)
        sigma_per_pos = (sigma_max * eta_conf * n_decoded).clamp(0.0, 1.0)

        # --- Step 5: Stochastic remasking ---
        remask_rand = torch.rand(batch_size, seq_len, device=device)
        return (remask_rand < sigma_per_pos) & remask_candidates


def create_remasking_schedule(
    config: dict,
    noise_schedule: NoiseSchedule | None,  # None allowed when rate_network provided
    vocab_config: VocabConfig,
    rate_network: torch.nn.Module | None = None,  # NEW
) -> RemaskingSchedule | None:
    """Factory: create a RemaskingSchedule from a config dict, or None if disabled.

    Args:
        config: Dict-like with keys "enabled" (bool), "strategy" (str), "eta" (float).
        noise_schedule: NoiseSchedule for alpha(t). None when using rate_network.
        vocab_config: VocabConfig for sequence layout.
        rate_network: Optional rate network for v2 learned forward process.

    Returns:
        RemaskingSchedule if enabled, else None.
    """
    if not config.get("enabled", False):
        return None
    return RemaskingSchedule(
        strategy=config["strategy"],
        eta=config["eta"],
        noise_schedule=noise_schedule,
        vocab_config=vocab_config,
        rate_network=rate_network,  # NEW
    )
