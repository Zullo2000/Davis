"""Learnable per-position rate network for the v2 forward process.

Maps timestep t in [0, 1] to per-position keeping probabilities alpha_l(t)
via learnable polynomial schedules. Each sequence position (8 nodes + 28
edges for RPLAN) gets its own monotonic schedule parameterised by a small
MLP over structural embeddings.

Mathematical summary
--------------------
For each position l:
  1. Learnable element embedding h_l (nodes: direct, edges: sum of endpoints).
  2. Type-specific linear projection proj_l.
  3. MLP -> softplus -> positive polynomial coefficients w^l in R^K.
  4. Polynomial evaluation:
       gamma_hat_l(t) = sum_k(w_k * t^k) / sum_k(w_k)   (weighted monomial avg)
       Properties: gamma_hat(0)=0, gamma_hat(1)=1, monotonically increasing.
  5. Scale: gamma_l(t) = gamma_hat_l(t) * (gamma_max - gamma_min) + gamma_min.
  6. Sigmoid: alpha_l(t) = sigmoid(-gamma_l(t)).
  7. Analytical derivative: alpha_prime_l(t) = -alpha*(1-alpha) * d_gamma/dt.

PAD positions always receive alpha=1.0 and alpha_prime=0.0.
"""

from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bd_gen.data.vocab import VocabConfig


class RateNetworkOutput(TypedDict):
    alpha: Tensor        # (B, SEQ_LEN) per-position keeping probability
    alpha_prime: Tensor  # (B, SEQ_LEN) per-position d(alpha)/dt
    gamma: Tensor        # (B, SEQ_LEN) pre-sigmoid log-odds (for debugging)


class RateNetwork(nn.Module):
    """Learnable per-position noise schedule for masked diffusion.

    Each of the SEQ_LEN positions in the flattened graph sequence gets
    its own monotonic schedule alpha_l(t). The schedule is parameterised
    by a polynomial whose coefficients are predicted from learnable
    structural embeddings (node identity for node positions, sum of
    endpoint embeddings for edge positions).

    Args:
        vocab_config: VocabConfig providing n_max, n_edges, seq_len, and
            edge_position_to_pair() for structural indexing.
        d_emb: Embedding dimension for the element embeddings. Default 32.
        K: Number of polynomial terms (monomials t, t^2, ..., t^K).
            Higher K allows more expressive schedules. Default 4.
        gamma_min: Log-odds at t=0 (clean). sigmoid(-gamma_min) ~ 1.
            Default -13.0.
        gamma_max: Log-odds at t=1 (masked). sigmoid(-gamma_max) ~ 0.
            Default 5.0.
        hidden_dim: Hidden dimension of the coefficient MLP. Default 64.
    """

    def __init__(
        self,
        vocab_config: VocabConfig,
        d_emb: int = 32,
        K: int = 4,
        gamma_min: float = -13.0,
        gamma_max: float = 5.0,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.vocab_config = vocab_config
        self.K = K

        # Store gamma bounds as buffers (move with .to(device))
        self.register_buffer(
            "gamma_min", torch.tensor(gamma_min, dtype=torch.float32)
        )
        self.register_buffer(
            "gamma_max", torch.tensor(gamma_max, dtype=torch.float32)
        )

        n_max = vocab_config.n_max
        n_edges = vocab_config.n_edges

        # --- Learnable element embeddings ---
        self.node_embeddings = nn.Embedding(n_max, d_emb)

        # Precompute edge endpoint indices (same pattern as
        # CompositePositionalEncoding in embeddings.py)
        edge_i = torch.zeros(n_edges, dtype=torch.long)
        edge_j = torch.zeros(n_edges, dtype=torch.long)
        for pos in range(n_edges):
            i, j = vocab_config.edge_position_to_pair(pos)
            edge_i[pos] = i
            edge_j[pos] = j
        self.register_buffer("edge_i", edge_i)
        self.register_buffer("edge_j", edge_j)

        # Precompute node indices for embedding lookup
        self.register_buffer(
            "node_indices", torch.arange(n_max, dtype=torch.long)
        )

        # --- Type-specific projections ---
        self.proj_node = nn.Linear(d_emb, d_emb)
        self.proj_edge = nn.Linear(d_emb, d_emb)

        # --- MLP: projected embedding -> K polynomial coefficients ---
        self.coeff_mlp = nn.Sequential(
            nn.Linear(d_emb, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, K),
        )

    def _compute_embeddings(self) -> Tensor:
        """Compute structural embeddings for all SEQ_LEN positions.

        Returns:
            (SEQ_LEN, d_emb) float tensor of projected embeddings.
        """
        n_max = self.vocab_config.n_max

        # Node embeddings: h_node^i for i in [0, n_max)
        h_nodes = self.node_embeddings(self.node_indices)  # (n_max, d_emb)

        # Edge embeddings: h_edge^{ij} = h_node^i + h_node^j
        h_edges = (
            self.node_embeddings(self.edge_i)
            + self.node_embeddings(self.edge_j)
        )  # (n_edges, d_emb)

        # Type-specific projection
        proj_nodes = self.proj_node(h_nodes)   # (n_max, d_emb)
        proj_edges = self.proj_edge(h_edges)   # (n_edges, d_emb)

        # Concatenate: [node_0, ..., node_{n_max-1}, edge_0, ..., edge_{n_edges-1}]
        return torch.cat([proj_nodes, proj_edges], dim=0)  # (SEQ_LEN, d_emb)

    def _compute_coefficients(self) -> Tensor:
        """Compute positive polynomial coefficients for all positions.

        Returns:
            (SEQ_LEN, K) float tensor of strictly positive coefficients.
        """
        proj = self._compute_embeddings()  # (SEQ_LEN, d_emb)
        raw = self.coeff_mlp(proj)         # (SEQ_LEN, K)
        return F.softplus(raw)             # (SEQ_LEN, K), strictly positive

    def _evaluate_polynomial(
        self,
        t: Tensor,
        w: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate the polynomial gamma_hat and its derivative.

        Args:
            t: (B,) float tensor of timesteps in [0, 1].
            w: (SEQ_LEN, K) positive polynomial coefficients.

        Returns:
            Tuple of:
                gamma_hat: (B, SEQ_LEN) values in [0, 1].
                d_gamma_hat_dt: (B, SEQ_LEN) derivative of gamma_hat w.r.t. t.
        """
        K = self.K
        # Exponents: k = 1, 2, ..., K
        exponents = torch.arange(1, K + 1, dtype=t.dtype, device=t.device)

        # t^k for each k: (B, K)
        # t: (B,) -> (B, 1), exponents: (K,) -> (1, K)
        t_powers = t.unsqueeze(1).pow(exponents.unsqueeze(0))  # (B, K)

        # Normalisation: sum of weights per position -> (SEQ_LEN,)
        w_sum = w.sum(dim=1)  # (SEQ_LEN,)

        # Numerator: sum_k(w_k * t^k) -> (B, SEQ_LEN)
        # w: (SEQ_LEN, K), t_powers: (B, K) -> matmul (B, SEQ_LEN)
        numerator = t_powers @ w.T  # (B, SEQ_LEN)

        # gamma_hat = numerator / w_sum, broadcast (SEQ_LEN,) over batch
        gamma_hat = numerator / w_sum.unsqueeze(0)  # (B, SEQ_LEN)

        # Derivative: d_gamma_hat/dt = sum_k(w_k * k * t^{k-1}) / w_sum
        # k * t^{k-1}: exponents * t^{exponents-1}
        # For k=1: 1 * t^0 = 1; for k=2: 2*t; etc.
        t_deriv_powers = exponents.unsqueeze(0) * t.unsqueeze(1).pow(
            (exponents - 1).unsqueeze(0)
        )  # (B, K)
        d_numerator = t_deriv_powers @ w.T  # (B, SEQ_LEN)
        d_gamma_hat_dt = d_numerator / w_sum.unsqueeze(0)  # (B, SEQ_LEN)

        return gamma_hat, d_gamma_hat_dt

    def forward(self, t: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        """Compute per-position alpha_l(t).

        Args:
            t: (B,) float32 tensor of timesteps in [0, 1].
            pad_mask: (B, SEQ_LEN) bool tensor. True = real position,
                False = PAD position. If None, all positions are treated
                as real.

        Returns:
            (B, SEQ_LEN) float32 tensor of keeping probabilities.
            PAD positions get alpha=1.0 (never masked).
        """
        output = self.forward_with_derivative(t, pad_mask)
        return output["alpha"]

    def alpha_prime(self, t: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        """Compute per-position d(alpha_l)/dt analytically.

        Args:
            t: (B,) float32 tensor of timesteps in [0, 1].
            pad_mask: (B, SEQ_LEN) bool tensor. True = real position.
                If None, all positions are treated as real.

        Returns:
            (B, SEQ_LEN) float32 tensor of alpha derivatives.
            PAD positions return 0.0.
        """
        output = self.forward_with_derivative(t, pad_mask)
        return output["alpha_prime"]

    def forward_with_derivative(
        self,
        t: Tensor,
        pad_mask: Tensor | None = None,
    ) -> RateNetworkOutput:
        """Compute alpha and alpha_prime in one efficient pass.

        Shares the polynomial evaluation between alpha and its derivative,
        avoiding redundant computation when both are needed (e.g. in the
        ELBO loss).

        Args:
            t: (B,) float32 tensor of timesteps in [0, 1].
            pad_mask: (B, SEQ_LEN) bool tensor. True = real position.
                If None, all positions are treated as real.

        Returns:
            RateNetworkOutput with keys:
                alpha: (B, SEQ_LEN) keeping probabilities.
                alpha_prime: (B, SEQ_LEN) d(alpha)/dt.
                gamma: (B, SEQ_LEN) pre-sigmoid log-odds (for debugging).
        """
        # Positive polynomial coefficients: (SEQ_LEN, K)
        w = self._compute_coefficients()

        # Polynomial and its derivative: both (B, SEQ_LEN)
        gamma_hat, d_gamma_hat_dt = self._evaluate_polynomial(t, w)

        # Scale to gamma range
        gamma_range = self.gamma_max - self.gamma_min
        gamma = gamma_hat * gamma_range + self.gamma_min  # (B, SEQ_LEN)

        # Sigmoid: alpha = sigmoid(-gamma)
        alpha = torch.sigmoid(-gamma)  # (B, SEQ_LEN)

        # Analytical derivative:
        #   d_gamma/dt = gamma_range * d_gamma_hat/dt
        #   d_alpha/dt = -alpha * (1 - alpha) * d_gamma/dt
        d_gamma_dt = gamma_range * d_gamma_hat_dt  # (B, SEQ_LEN)
        alpha_prime = -alpha * (1.0 - alpha) * d_gamma_dt  # (B, SEQ_LEN)

        # PAD handling: force alpha=1.0 and alpha_prime=0.0 for PAD positions
        if pad_mask is not None:
            pad_positions = ~pad_mask  # True where PAD
            alpha = alpha.masked_fill(pad_positions, 1.0)
            alpha_prime = alpha_prime.masked_fill(pad_positions, 0.0)
            gamma = gamma.masked_fill(pad_positions, self.gamma_min.item())

        return RateNetworkOutput(
            alpha=alpha,
            alpha_prime=alpha_prime,
            gamma=gamma,
        )
