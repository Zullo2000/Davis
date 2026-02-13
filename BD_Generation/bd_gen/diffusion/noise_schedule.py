"""Noise schedules for MDLM masked diffusion.

Provides continuous-time noise schedules that map t in [0, 1] to a masking
probability. All schedules compute sigma(t) (total noise) and derive
alpha(t) = exp(-sigma(t)) (keeping probability). At t=0 tokens are clean
(alpha ~ 1); at t=1 tokens are fully masked (alpha ~ 0).

Factory function get_noise(config) returns the appropriate schedule from
a Hydra/OmegaConf config object.

Mathematical reference: MDLM (Sahoo et al., arXiv:2406.07524).
"""

from __future__ import annotations

import abc
import math

import torch
import torch.nn as nn
from torch import Tensor


class NoiseSchedule(abc.ABC, nn.Module):
    """Abstract base for noise schedules.

    Subclasses implement sigma(t) (total noise). The base class derives
    alpha(t) = exp(-sigma(t)) and provides a default (identity)
    importance_sampling_transformation.

    All methods accept and return float32 tensors. Input t can be any shape;
    output matches the input shape (element-wise operations only).
    """

    @abc.abstractmethod
    def sigma(self, t: Tensor) -> Tensor:
        """Total noise at time t."""
        ...

    def alpha(self, t: Tensor) -> Tensor:
        """Keeping probability: alpha_t = exp(-sigma(t)).

        Returns probability that a token is NOT masked at time t.
        """
        return torch.exp(-self.sigma(t))

    @abc.abstractmethod
    def alpha_prime(self, t: Tensor) -> Tensor:
        """Derivative d(alpha_t)/dt. Needed for ELBO loss weight."""
        ...

    def importance_sampling_transformation(self, t: Tensor) -> Tensor:
        """Map uniform t to importance-weighted t (identity by default)."""
        return t


class LinearSchedule(NoiseSchedule):
    """Linear noise schedule: sigma_t = sigma_min + t * (sigma_max - sigma_min).

    With default sigma_min=0, sigma_max=10:
      - t=0: sigma=0, alpha=1 (clean)
      - t=1: sigma=10, alpha=exp(-10) ~ 4.5e-5 (fully masked)

    Args:
        sigma_min: Minimum noise level (at t=0). Default 0.0.
        sigma_max: Maximum noise level (at t=1). Default 10.0.
    """

    def __init__(self, sigma_min: float = 0.0, sigma_max: float = 10.0) -> None:
        super().__init__()
        self.register_buffer(
            "sigma_min", torch.tensor(sigma_min, dtype=torch.float32)
        )
        self.register_buffer(
            "sigma_max", torch.tensor(sigma_max, dtype=torch.float32)
        )

    def sigma(self, t: Tensor) -> Tensor:
        return self.sigma_min + t * (self.sigma_max - self.sigma_min)

    def alpha_prime(self, t: Tensor) -> Tensor:
        """d(alpha_t)/dt = -(sigma_max - sigma_min) * exp(-sigma_t)."""
        return -(self.sigma_max - self.sigma_min) * self.alpha(t)

    def importance_sampling_transformation(self, t: Tensor) -> Tensor:
        """MDLM importance sampling for linear schedule.

        Maps uniform t to a distribution that reduces ELBO gradient variance.
        Adapted from MDLM / DiDAPS reference implementation.

        When sigma_min=0, log1p(-exp(0)) = -inf which causes NaN via 0*-inf.
        We clamp sigma_min to a small positive value to avoid this singularity.
        """
        # Clamp to avoid log(0) when sigma_min == 0
        s_min = torch.clamp(self.sigma_min, min=1e-4)
        s_max = self.sigma_max
        f_T = torch.log1p(-torch.exp(-s_max))
        f_0 = torch.log1p(-torch.exp(-s_min))
        sigma_t = -torch.log1p(-torch.exp(t * f_T + (1 - t) * f_0))
        return (sigma_t - self.sigma_min) / (self.sigma_max - self.sigma_min)


class CosineSchedule(NoiseSchedule):
    """Cosine noise schedule: alpha_t = eps + (1 - eps) * cos(t * pi/2).

    Directly defines alpha_t (no log-then-exp), giving a smoother masking
    curve that spends more time near intermediate noise levels.

    Args:
        eps: Small constant to prevent alpha from reaching exactly 0.
            Default 1e-3.
    """

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.register_buffer("eps", torch.tensor(eps, dtype=torch.float32))

    def sigma(self, t: Tensor) -> Tensor:
        """sigma_t = -log(alpha_t), with clamping for numerical safety."""
        alpha_t = self.alpha(t)
        return -torch.log(torch.clamp(alpha_t, min=1e-8))

    def alpha(self, t: Tensor) -> Tensor:
        """Direct formula avoids exp(-(-log(x))) numerical roundtrip."""
        return self.eps + (1 - self.eps) * torch.cos(t * math.pi / 2)

    def alpha_prime(self, t: Tensor) -> Tensor:
        """d(alpha_t)/dt = -(1 - eps) * (pi/2) * sin(t * pi/2)."""
        return -(1 - self.eps) * (math.pi / 2) * torch.sin(t * math.pi / 2)


def get_noise(config) -> NoiseSchedule:
    """Create a noise schedule from a config object.

    Args:
        config: An object with a ``type`` attribute. For linear schedules,
            also requires ``sigma_min`` and ``sigma_max``. For cosine,
            requires ``eps``.

    Returns:
        A NoiseSchedule instance.

    Raises:
        ValueError: If config.type is not recognized.
    """
    if config.type == "linear":
        return LinearSchedule(
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
        )
    elif config.type == "cosine":
        return CosineSchedule(eps=config.eps)
    else:
        raise ValueError(
            f"Unknown noise schedule type: '{config.type}'. "
            f"Supported: 'linear', 'cosine'."
        )
