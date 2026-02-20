"""Tests for the learnable per-position rate network.

Covers:
- Output shapes and dtypes
- Boundary conditions at t=0 (clean) and t=1 (masked)
- Monotonicity of alpha(t) over time
- PAD invariants (alpha=1, alpha_prime=0)
- Sign of alpha_prime (non-positive for real positions)
- Gradient flow through learnable embeddings
- Position diversity after gradient updates
- Consistency between forward, alpha_prime, and forward_with_derivative
"""

from __future__ import annotations

import pytest
import torch

from bd_gen.data.vocab import RPLAN_VOCAB_CONFIG, VocabConfig
from bd_gen.diffusion.rate_network import RateNetwork


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vc() -> VocabConfig:
    """Return the RPLAN VocabConfig preset."""
    return RPLAN_VOCAB_CONFIG


@pytest.fixture
def rate_net(vc: VocabConfig) -> RateNetwork:
    """Small RateNetwork with default parameters for fast tests."""
    torch.manual_seed(42)
    return RateNetwork(vocab_config=vc, d_emb=32, K=4, hidden_dim=64)


@pytest.fixture
def sample_t() -> torch.Tensor:
    """Batch of 4 timesteps spanning [0, 1]."""
    return torch.tensor([0.0, 0.3, 0.7, 1.0], dtype=torch.float32)


@pytest.fixture
def pad_mask(vc: VocabConfig) -> torch.Tensor:
    """Batch of 4 pad masks with varying num_rooms: 2, 4, 6, 8."""
    num_rooms_list = [2, 4, 6, 8]
    masks = torch.stack(
        [vc.compute_pad_mask(nr) for nr in num_rooms_list], dim=0
    )
    return masks


# ---------------------------------------------------------------------------
# Test 1: Shape check
# ---------------------------------------------------------------------------


class TestShapes:
    """Verify output tensor shapes match (B, SEQ_LEN)."""

    def test_forward_shape(
        self, rate_net: RateNetwork, sample_t: torch.Tensor,
        pad_mask: torch.Tensor, vc: VocabConfig,
    ) -> None:
        alpha = rate_net(sample_t, pad_mask)
        B = sample_t.shape[0]
        assert alpha.shape == (B, vc.seq_len)

    def test_alpha_prime_shape(
        self, rate_net: RateNetwork, sample_t: torch.Tensor,
        pad_mask: torch.Tensor, vc: VocabConfig,
    ) -> None:
        ap = rate_net.alpha_prime(sample_t, pad_mask)
        B = sample_t.shape[0]
        assert ap.shape == (B, vc.seq_len)

    def test_forward_with_derivative_shapes(
        self, rate_net: RateNetwork, sample_t: torch.Tensor,
        pad_mask: torch.Tensor, vc: VocabConfig,
    ) -> None:
        out = rate_net.forward_with_derivative(sample_t, pad_mask)
        B = sample_t.shape[0]
        assert out["alpha"].shape == (B, vc.seq_len)
        assert out["alpha_prime"].shape == (B, vc.seq_len)
        assert out["gamma"].shape == (B, vc.seq_len)


# ---------------------------------------------------------------------------
# Test 2: Boundary t=0 -> alpha > 0.99
# ---------------------------------------------------------------------------


class TestBoundaryT0:
    """At t=0, all positions should be nearly clean (alpha close to 1)."""

    def test_alpha_near_one_at_t0(
        self, rate_net: RateNetwork, vc: VocabConfig,
    ) -> None:
        t = torch.tensor([0.0, 0.0, 0.0, 0.0])
        # No pad_mask: all positions are real
        alpha = rate_net(t)
        assert (alpha > 0.99).all(), (
            f"Expected all alpha > 0.99 at t=0, got min={alpha.min().item():.6f}"
        )


# ---------------------------------------------------------------------------
# Test 3: Boundary t=1 -> alpha < 0.02 for real positions
# ---------------------------------------------------------------------------


class TestBoundaryT1:
    """At t=1, real positions should be nearly fully masked (alpha near 0)."""

    def test_alpha_near_zero_at_t1(
        self, rate_net: RateNetwork, pad_mask: torch.Tensor,
    ) -> None:
        B = pad_mask.shape[0]
        t = torch.ones(B)
        alpha = rate_net(t, pad_mask)
        # Check only real (non-PAD) positions
        real_alphas = alpha[pad_mask]
        assert (real_alphas < 0.02).all(), (
            f"Expected real alpha < 0.02 at t=1, got max={real_alphas.max().item():.6f}"
        )


# ---------------------------------------------------------------------------
# Test 4: Monotonicity
# ---------------------------------------------------------------------------


class TestMonotonicity:
    """For t1 < t2, alpha(t1) > alpha(t2) element-wise (on real positions)."""

    def test_alpha_decreasing(
        self, rate_net: RateNetwork, vc: VocabConfig,
    ) -> None:
        # Dense grid of timesteps
        n_steps = 50
        t_values = torch.linspace(0.0, 1.0, n_steps)

        # Evaluate alpha at each timestep (batch dim = 1 per call)
        alphas = []
        for t_val in t_values:
            t_batch = t_val.unsqueeze(0)  # (1,)
            alpha = rate_net(t_batch)      # (1, SEQ_LEN)
            alphas.append(alpha.squeeze(0))

        alphas = torch.stack(alphas, dim=0)  # (n_steps, SEQ_LEN)

        # Check monotonic decrease: alpha[i] >= alpha[i+1] for all i
        diffs = alphas[1:] - alphas[:-1]  # (n_steps-1, SEQ_LEN)
        assert (diffs <= 1e-5).all(), (
            f"Monotonicity violated: max increase = {diffs.max().item():.6f}"
        )


# ---------------------------------------------------------------------------
# Test 5: PAD invariant
# ---------------------------------------------------------------------------


class TestPadInvariant:
    """PAD positions must have alpha == 1.0 exactly."""

    def test_pad_alpha_equals_one(
        self, rate_net: RateNetwork, pad_mask: torch.Tensor,
    ) -> None:
        B = pad_mask.shape[0]
        t = torch.tensor([0.1, 0.4, 0.7, 0.95])
        alpha = rate_net(t, pad_mask)
        pad_alphas = alpha[~pad_mask]
        assert (pad_alphas == 1.0).all(), (
            f"PAD positions should have alpha=1.0, got values != 1.0"
        )


# ---------------------------------------------------------------------------
# Test 6: alpha_prime sign (non-positive for real positions)
# ---------------------------------------------------------------------------


class TestAlphaPrimeSign:
    """alpha_prime should be <= 0 everywhere for real positions."""

    def test_alpha_prime_nonpositive(
        self, rate_net: RateNetwork, pad_mask: torch.Tensor,
    ) -> None:
        B = pad_mask.shape[0]
        # Test at several interior timesteps
        t = torch.tensor([0.1, 0.3, 0.6, 0.9])
        ap = rate_net.alpha_prime(t, pad_mask)
        real_ap = ap[pad_mask]
        assert (real_ap <= 1e-6).all(), (
            f"alpha_prime should be <= 0 for real positions, "
            f"got max={real_ap.max().item():.6f}"
        )


# ---------------------------------------------------------------------------
# Test 7: alpha_prime PAD
# ---------------------------------------------------------------------------


class TestAlphaPrimePad:
    """alpha_prime for PAD positions must be exactly 0.0."""

    def test_pad_alpha_prime_zero(
        self, rate_net: RateNetwork, pad_mask: torch.Tensor,
    ) -> None:
        B = pad_mask.shape[0]
        t = torch.tensor([0.2, 0.5, 0.8, 0.99])
        ap = rate_net.alpha_prime(t, pad_mask)
        pad_ap = ap[~pad_mask]
        assert (pad_ap == 0.0).all(), (
            f"PAD positions should have alpha_prime=0.0"
        )


# ---------------------------------------------------------------------------
# Test 8: Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Backward pass should produce gradients on node_embeddings.weight."""

    def test_grad_on_node_embeddings(
        self, rate_net: RateNetwork, pad_mask: torch.Tensor,
    ) -> None:
        t = torch.tensor([0.2, 0.5, 0.7, 0.9])
        alpha = rate_net(t, pad_mask)
        # Use a simple loss that touches all positions
        loss = alpha[pad_mask].sum()
        loss.backward()
        assert rate_net.node_embeddings.weight.grad is not None, (
            "node_embeddings.weight.grad should not be None after backward"
        )
        assert (rate_net.node_embeddings.weight.grad.abs() > 0).any(), (
            "node_embeddings.weight.grad should have non-zero entries"
        )


# ---------------------------------------------------------------------------
# Test 9: Different positions
# ---------------------------------------------------------------------------


class TestPositionDiversity:
    """Not all positions should have identical alpha after gradient steps.

    Initially the embeddings might produce similar outputs for all
    positions; after a few gradient steps optimising different positions
    differently, the schedules should diverge.
    """

    def test_positions_diverge_after_training(
        self, vc: VocabConfig,
    ) -> None:
        torch.manual_seed(123)
        net = RateNetwork(vocab_config=vc, d_emb=32, K=4, hidden_dim=64)
        optimiser = torch.optim.Adam(net.parameters(), lr=5e-2)

        # Create per-position targets at multiple timesteps to provide a
        # strong and diverse gradient signal. Node positions target high
        # alpha, edge positions target low alpha -- this should force the
        # learned embeddings to differentiate.
        t_vals = torch.tensor([0.3, 0.5, 0.7])
        target = torch.zeros(3, vc.seq_len)
        target[:, :vc.n_max] = 0.85   # nodes: keep high
        target[:, vc.n_max:] = 0.15    # edges: keep low

        for _ in range(200):
            optimiser.zero_grad()
            alpha = net(t_vals)
            loss = ((alpha - target) ** 2).sum()
            loss.backward()
            optimiser.step()

        # After training, node and edge positions should have diverged.
        # Check at t=0.5 where the sigmoid is most responsive.
        t_test = torch.tensor([0.5])
        alpha_final = net(t_test).detach().squeeze(0)  # (SEQ_LEN,)
        node_mean = alpha_final[:vc.n_max].mean()
        edge_mean = alpha_final[vc.n_max:].mean()
        assert abs(node_mean.item() - edge_mean.item()) > 0.01, (
            f"Node and edge positions should differ after training, "
            f"got node_mean={node_mean.item():.4f}, edge_mean={edge_mean.item():.4f}"
        )


# ---------------------------------------------------------------------------
# Test 10: Consistency between forward, alpha_prime, forward_with_derivative
# ---------------------------------------------------------------------------


class TestConsistency:
    """forward_with_derivative() must match forward() + alpha_prime()."""

    def test_combined_matches_separate(
        self, rate_net: RateNetwork, pad_mask: torch.Tensor,
    ) -> None:
        t = torch.tensor([0.15, 0.45, 0.75, 0.95])

        # Separate calls
        alpha_separate = rate_net(t, pad_mask)
        ap_separate = rate_net.alpha_prime(t, pad_mask)

        # Combined call
        out = rate_net.forward_with_derivative(t, pad_mask)
        alpha_combined = out["alpha"]
        ap_combined = out["alpha_prime"]

        assert torch.allclose(alpha_separate, alpha_combined, atol=1e-6), (
            "alpha from forward() and forward_with_derivative() differ"
        )
        assert torch.allclose(ap_separate, ap_combined, atol=1e-6), (
            "alpha_prime from alpha_prime() and forward_with_derivative() differ"
        )
