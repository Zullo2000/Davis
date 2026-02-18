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
        - "rescale": sigma_t = eta * sigma_max.

    sigma_t is the probability that an already-unmasked, non-PAD position
    gets re-masked at each denoising step.

    Args:
        strategy: Remasking strategy ("cap" or "rescale").
        eta: Remasking intensity. 0 = no remasking, higher = more aggressive.
        noise_schedule: NoiseSchedule for alpha(t) computation.
        vocab_config: VocabConfig for n_max (node/edge position split).
    """

    def __init__(
        self,
        strategy: str,
        eta: float,
        noise_schedule: NoiseSchedule,
        vocab_config: VocabConfig,
    ) -> None:
        if strategy not in ("cap", "rescale"):
            raise ValueError(
                f"Unknown remasking strategy: {strategy!r}. Use 'cap' or 'rescale'."
            )
        if eta < 0:
            raise ValueError(f"eta must be non-negative, got {eta}")
        self.strategy = strategy
        self.eta = eta
        self.noise_schedule = noise_schedule
        self.vocab_config = vocab_config

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
        t_now_tensor = torch.full(
            (batch_size,), t_now, dtype=torch.float64, device=device
        )
        t_next_tensor = torch.full(
            (batch_size,), t_next, dtype=torch.float64, device=device
        )

        alpha_t = self.noise_schedule.alpha(t_now_tensor)  # (B,) float64
        alpha_s = self.noise_schedule.alpha(t_next_tensor)  # (B,) float64

        # Upper bound: sigma_max = min(1, (1 - alpha_s) / alpha_t)
        sigma_max = (1.0 - alpha_s) / (alpha_t + 1e-8)
        sigma_max = torch.clamp(sigma_max, min=0.0, max=1.0)

        if self.strategy == "cap":
            sigma_t = torch.minimum(
                torch.tensor(self.eta, dtype=torch.float64, device=device),
                sigma_max,
            )
        else:  # rescale
            sigma_t = self.eta * sigma_max

        return sigma_t.float().unsqueeze(1)  # (B, 1)

    def __call__(
        self,
        x_t: Tensor,
        t_now: float,
        t_next: float,
        pad_mask: Tensor,
    ) -> Tensor:
        """Apply stochastic remasking to already-decoded positions.

        Args:
            x_t: (B, SEQ_LEN) long tensor of current tokens.
            t_now: Current timestep (noisier).
            t_next: Destination timestep (cleaner).
            pad_mask: (B, SEQ_LEN) bool tensor, True=real position.

        Returns:
            (B, SEQ_LEN) long tensor with some positions re-masked.
        """
        if self.eta == 0.0:
            return x_t

        batch_size, seq_len = x_t.shape
        n_max = self.vocab_config.n_max
        device = x_t.device

        sigma_t = self._compute_sigma_t(t_now, t_next, batch_size, device)

        # Identify positions that are NOT currently masked (candidates for remasking)
        is_node_mask = x_t[:, :n_max] == NODE_MASK_IDX
        is_edge_mask = x_t[:, n_max:] == EDGE_MASK_IDX
        is_mask = torch.cat([is_node_mask, is_edge_mask], dim=1)

        # Remask candidates: non-MASK AND non-PAD (never remask PAD)
        remask_candidates = (~is_mask) & pad_mask

        # Stochastic remasking decision
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


def create_remasking_schedule(
    config: dict,
    noise_schedule: NoiseSchedule,
    vocab_config: VocabConfig,
) -> RemaskingSchedule | None:
    """Factory: create a RemaskingSchedule from a config dict, or None if disabled.

    Args:
        config: Dict-like with keys "enabled" (bool), "strategy" (str), "eta" (float).
        noise_schedule: NoiseSchedule for alpha(t) computation.
        vocab_config: VocabConfig for sequence layout.

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
    )
