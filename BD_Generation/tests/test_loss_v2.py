"""Tests for ELBOLossV2 (learned forward process per-position ELBO loss).

Covers spec Section 14.4 tests 24-30:
24. Scalar output: loss.shape == ()
25. Positive loss: loss > 0 for random logits
26. Gradient to denoiser: model parameters have gradients after backward
27. Gradient to rate net: rate_network.node_embeddings.weight.grad is not None
28. PAD exclusion: Loss unchanged when PAD-position logits are randomized
29. Lambda=0 edges: lambda_edge=0 -> loss depends only on node positions
30. Separate normalization: N_active_nodes and N_active_edges computed independently
"""

from __future__ import annotations

import pytest
import torch

from bd_gen.data.vocab import (
    EDGE_PAD_IDX,
    EDGE_VOCAB_SIZE,
    NODE_PAD_IDX,
    NODE_VOCAB_SIZE,
    RPLAN_VOCAB_CONFIG,
)
from bd_gen.diffusion.loss import ELBOLossV2
from bd_gen.diffusion.rate_network import RateNetwork
from bd_gen.model.denoiser import BDDenoiser


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vc():
    """RPLAN VocabConfig."""
    return RPLAN_VOCAB_CONFIG


@pytest.fixture
def edge_weights():
    """Uniform edge class weights for testing."""
    return torch.ones(EDGE_VOCAB_SIZE)


@pytest.fixture
def loss_fn(edge_weights, vc):
    """ELBOLossV2 with default settings."""
    return ELBOLossV2(
        edge_class_weights=edge_weights,
        node_class_weights=None,
        vocab_config=vc,
    )


@pytest.fixture
def loss_fn_lambda0(edge_weights, vc):
    """ELBOLossV2 with lambda_edge=0."""
    return ELBOLossV2(
        edge_class_weights=edge_weights,
        node_class_weights=None,
        vocab_config=vc,
        lambda_edge=0.0,
    )


@pytest.fixture
def rate_net(vc):
    """Small RateNetwork for testing."""
    return RateNetwork(vocab_config=vc, d_emb=16, K=3, hidden_dim=32)


@pytest.fixture
def denoiser(vc):
    """Small BDDenoiser for testing."""
    return BDDenoiser(
        d_model=32,
        n_layers=1,
        n_heads=2,
        vocab_config=vc,
        dropout=0.0,
    )


@pytest.fixture
def sample_inputs(vc):
    """Create a complete set of v2 loss inputs.

    Returns a dict with all tensors needed for ELBOLossV2.forward().
    Uses num_rooms=[3, 5, 8] as a mixed batch.
    """
    torch.manual_seed(42)
    num_rooms_list = [3, 5, 8]
    B = len(num_rooms_list)

    tokens = torch.zeros(B, vc.seq_len, dtype=torch.long)
    pad_mask = torch.zeros(B, vc.seq_len, dtype=torch.bool)

    for b, nr in enumerate(num_rooms_list):
        mask = vc.compute_pad_mask(nr)
        pad_mask[b] = mask
        for k in range(nr):
            tokens[b, k] = torch.randint(0, 13, (1,))
        for k in range(nr, vc.n_max):
            tokens[b, k] = NODE_PAD_IDX
        for pos in range(vc.n_edges):
            seq_idx = vc.n_max + pos
            if mask[seq_idx]:
                tokens[b, seq_idx] = torch.randint(0, 11, (1,))
            else:
                tokens[b, seq_idx] = EDGE_PAD_IDX

    # Simulate mid-diffusion: mask ~50% of non-PAD positions
    rand = torch.rand(B, vc.seq_len)
    mask_indicators = (rand > 0.5) & pad_mask

    # Simulate rate network outputs: alpha in (0,1), alpha_prime < 0
    alpha_per_pos = torch.rand(B, vc.seq_len) * 0.8 + 0.1  # (0.1, 0.9)
    alpha_prime_per_pos = -torch.rand(B, vc.seq_len) * 2.0 - 0.1  # negative

    # PAD positions: alpha=1.0, alpha_prime=0.0
    alpha_per_pos[~pad_mask] = 1.0
    alpha_prime_per_pos[~pad_mask] = 0.0

    node_logits = torch.randn(B, vc.n_max, NODE_VOCAB_SIZE, requires_grad=True)
    edge_logits = torch.randn(B, vc.n_edges, EDGE_VOCAB_SIZE, requires_grad=True)

    return {
        "node_logits": node_logits,
        "edge_logits": edge_logits,
        "x0": tokens,
        "pad_mask": pad_mask,
        "mask_indicators": mask_indicators,
        "alpha_per_pos": alpha_per_pos,
        "alpha_prime_per_pos": alpha_prime_per_pos,
    }


# =========================================================================
# Test 24: Scalar output
# =========================================================================


class TestScalarOutput:
    """Test 24: loss.shape == ()."""

    def test_loss_is_scalar(self, loss_fn, sample_inputs):
        loss = loss_fn(**sample_inputs)
        assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
        assert loss.dim() == 0


# =========================================================================
# Test 25: Positive loss
# =========================================================================


class TestPositiveLoss:
    """Test 25: loss > 0 for random logits."""

    def test_loss_positive(self, loss_fn, sample_inputs):
        loss = loss_fn(**sample_inputs)
        assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"

    def test_loss_finite(self, loss_fn, sample_inputs):
        loss = loss_fn(**sample_inputs)
        assert torch.isfinite(loss), f"Expected finite loss, got {loss.item()}"

    def test_loss_dtype(self, loss_fn, sample_inputs):
        loss = loss_fn(**sample_inputs)
        assert loss.dtype == torch.float32


# =========================================================================
# Test 26: Gradient to denoiser
# =========================================================================


class TestGradientToDenoiser:
    """Test 26: denoiser parameters have gradients after backward."""

    def test_denoiser_grads(self, loss_fn, denoiser, vc):
        torch.manual_seed(99)
        B = 2
        tokens = torch.randint(0, 13, (B, vc.seq_len))
        tokens[:, vc.n_max:] = torch.randint(0, 11, (B, vc.n_edges))
        pad_mask = torch.ones(B, vc.seq_len, dtype=torch.bool)

        # Mask ~50% of positions
        mask_indicators = torch.rand(B, vc.seq_len) > 0.5

        # Simulate rate network alpha values (detached, not part of denoiser)
        alpha_per_pos = torch.rand(B, vc.seq_len) * 0.8 + 0.1
        alpha_prime_per_pos = -torch.rand(B, vc.seq_len) * 2.0 - 0.1

        t = torch.tensor([0.5, 0.3])
        node_logits, edge_logits = denoiser(tokens, pad_mask, t)

        loss = loss_fn(
            node_logits, edge_logits, tokens, pad_mask,
            mask_indicators, alpha_per_pos, alpha_prime_per_pos,
        )
        loss.backward()

        # Check that at least some denoiser parameters have gradients
        has_grad = False
        for name, p in denoiser.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No denoiser parameter received a gradient"


# =========================================================================
# Test 27: Gradient to rate network
# =========================================================================


class TestGradientToRateNet:
    """Test 27: rate_network.node_embeddings.weight.grad is not None."""

    def test_rate_net_grads(self, loss_fn, rate_net, vc):
        torch.manual_seed(77)
        B = 2
        tokens = torch.randint(0, 13, (B, vc.seq_len))
        tokens[:, vc.n_max:] = torch.randint(0, 11, (B, vc.n_edges))
        pad_mask = torch.ones(B, vc.seq_len, dtype=torch.bool)

        t = torch.tensor([0.5, 0.3])
        rate_out = rate_net.forward_with_derivative(t, pad_mask)
        alpha_per_pos = rate_out["alpha"]
        alpha_prime_per_pos = rate_out["alpha_prime"]

        # Mask ~50% of positions
        mask_indicators = torch.rand(B, vc.seq_len) > 0.5

        node_logits = torch.randn(B, vc.n_max, NODE_VOCAB_SIZE, requires_grad=True)
        edge_logits = torch.randn(B, vc.n_edges, EDGE_VOCAB_SIZE, requires_grad=True)

        loss = loss_fn(
            node_logits, edge_logits, tokens, pad_mask,
            mask_indicators, alpha_per_pos, alpha_prime_per_pos,
        )
        loss.backward()

        assert rate_net.node_embeddings.weight.grad is not None, (
            "rate_network.node_embeddings.weight.grad is None"
        )
        assert rate_net.node_embeddings.weight.grad.abs().sum() > 0, (
            "rate_network.node_embeddings.weight.grad is all zeros"
        )


# =========================================================================
# Test 28: PAD exclusion
# =========================================================================


class TestPadExclusion:
    """Test 28: Loss unchanged when PAD-position logits are randomized."""

    def test_pad_logits_dont_affect_loss(self, loss_fn, vc):
        torch.manual_seed(55)
        # Use num_rooms=3 so most positions are PAD
        num_rooms = 3
        B = 1
        tokens = torch.zeros(B, vc.seq_len, dtype=torch.long)
        pad_mask = vc.compute_pad_mask(num_rooms).unsqueeze(0)

        for k in range(num_rooms):
            tokens[0, k] = torch.randint(0, 13, (1,))
        for k in range(num_rooms, vc.n_max):
            tokens[0, k] = NODE_PAD_IDX
        for pos in range(vc.n_edges):
            seq_idx = vc.n_max + pos
            if pad_mask[0, seq_idx]:
                tokens[0, seq_idx] = torch.randint(0, 11, (1,))
            else:
                tokens[0, seq_idx] = EDGE_PAD_IDX

        # Mask ~50% of non-PAD positions
        mask_indicators = (torch.rand(B, vc.seq_len) > 0.5) & pad_mask

        alpha_per_pos = torch.rand(B, vc.seq_len) * 0.8 + 0.1
        alpha_prime_per_pos = -torch.rand(B, vc.seq_len) * 2.0 - 0.1
        alpha_per_pos[~pad_mask] = 1.0
        alpha_prime_per_pos[~pad_mask] = 0.0

        node_logits = torch.randn(B, vc.n_max, NODE_VOCAB_SIZE)
        edge_logits = torch.randn(B, vc.n_edges, EDGE_VOCAB_SIZE)

        loss_1 = loss_fn(
            node_logits, edge_logits, tokens, pad_mask,
            mask_indicators, alpha_per_pos, alpha_prime_per_pos,
        )

        # Randomize logits at PAD positions
        node_logits_2 = node_logits.clone()
        edge_logits_2 = edge_logits.clone()
        # Node PAD positions: rooms 3-7
        node_logits_2[:, num_rooms:, :] = torch.randn(
            B, vc.n_max - num_rooms, NODE_VOCAB_SIZE
        ) * 100.0
        # Edge PAD positions
        for pos in range(vc.n_edges):
            i, j = vc.edge_position_to_pair(pos)
            if i >= num_rooms or j >= num_rooms:
                edge_logits_2[:, pos, :] = torch.randn(EDGE_VOCAB_SIZE) * 100.0

        loss_2 = loss_fn(
            node_logits_2, edge_logits_2, tokens, pad_mask,
            mask_indicators, alpha_per_pos, alpha_prime_per_pos,
        )

        assert torch.isclose(loss_1, loss_2, atol=1e-5), (
            f"PAD logit changes affected loss: {loss_1.item()} vs {loss_2.item()}"
        )


# =========================================================================
# Test 29: Lambda=0 edges
# =========================================================================


class TestLambdaZeroEdges:
    """Test 29: lambda_edge=0 -> loss depends only on node positions."""

    def test_lambda_zero_ignores_edges(self, loss_fn_lambda0, vc):
        torch.manual_seed(33)
        B = 2
        tokens = torch.randint(0, 13, (B, vc.seq_len))
        tokens[:, vc.n_max:] = torch.randint(0, 11, (B, vc.n_edges))
        pad_mask = torch.ones(B, vc.seq_len, dtype=torch.bool)
        mask_indicators = torch.rand(B, vc.seq_len) > 0.5

        alpha_per_pos = torch.rand(B, vc.seq_len) * 0.8 + 0.1
        alpha_prime_per_pos = -torch.rand(B, vc.seq_len) * 2.0 - 0.1

        node_logits = torch.randn(B, vc.n_max, NODE_VOCAB_SIZE)

        # Two different edge logits
        edge_logits_a = torch.randn(B, vc.n_edges, EDGE_VOCAB_SIZE)
        edge_logits_b = torch.randn(B, vc.n_edges, EDGE_VOCAB_SIZE) * 10.0

        loss_a = loss_fn_lambda0(
            node_logits, edge_logits_a, tokens, pad_mask,
            mask_indicators, alpha_per_pos, alpha_prime_per_pos,
        )
        loss_b = loss_fn_lambda0(
            node_logits, edge_logits_b, tokens, pad_mask,
            mask_indicators, alpha_per_pos, alpha_prime_per_pos,
        )

        assert torch.isclose(loss_a, loss_b, atol=1e-5), (
            f"With lambda_edge=0, edge logits should not affect loss: "
            f"{loss_a.item()} vs {loss_b.item()}"
        )


# =========================================================================
# Test 30: Separate normalization
# =========================================================================


class TestSeparateNormalization:
    """Test 30: N_active_nodes and N_active_edges computed independently.

    Verifies that the loss uses separate per-type normalization rather
    than a single combined N_active count. We construct a scenario where
    combined vs separate normalization gives different results.
    """

    def test_separate_node_edge_normalization(self, edge_weights, vc):
        """Construct inputs where separate vs combined normalization differs.

        Setup: 1 active node position, many active edge positions.
        With separate normalization: node_loss / 1 + edge_loss / N_edges.
        With combined normalization: (node_loss + edge_loss) / (1 + N_edges).
        These will differ unless node_loss == edge_loss (extremely unlikely).
        """
        torch.manual_seed(88)
        B = 1
        # Use num_rooms=8 (all positions real), but mask only 1 node and all edges
        pad_mask = vc.compute_pad_mask(8).unsqueeze(0)  # (1, 36)
        tokens = torch.zeros(B, vc.seq_len, dtype=torch.long)
        tokens[0, :vc.n_max] = torch.randint(0, 13, (vc.n_max,))
        tokens[0, vc.n_max:] = torch.randint(0, 11, (vc.n_edges,))

        # Mask only node 0, and all edges
        mask_indicators = torch.zeros(B, vc.seq_len, dtype=torch.bool)
        mask_indicators[0, 0] = True   # Only 1 node masked
        mask_indicators[0, vc.n_max:] = True  # All edges masked

        alpha_per_pos = torch.rand(B, vc.seq_len) * 0.8 + 0.1
        alpha_prime_per_pos = -torch.rand(B, vc.seq_len) * 2.0 - 0.1

        node_logits = torch.randn(B, vc.n_max, NODE_VOCAB_SIZE)
        edge_logits = torch.randn(B, vc.n_edges, EDGE_VOCAB_SIZE)

        # Compute with lambda_edge=1.0 (default)
        loss_fn = ELBOLossV2(
            edge_class_weights=edge_weights,
            vocab_config=vc,
            lambda_edge=1.0,
        )
        loss = loss_fn(
            node_logits, edge_logits, tokens, pad_mask,
            mask_indicators, alpha_per_pos, alpha_prime_per_pos,
        )

        # Manually compute what combined normalization would give
        # (to verify we are NOT using combined normalization)
        n_active_nodes = 1  # Only node 0 masked
        n_active_edges = vc.n_edges  # All 28 edges masked
        n_active_combined = n_active_nodes + n_active_edges

        # If separate: node_loss / 1 + edge_loss / 28
        # If combined: (node_loss + edge_loss) / 29
        # These are different unless node_loss == edge_loss (probability 0)

        # Verify by computing manually with separate normalization
        import torch.nn.functional as F

        n_max = vc.n_max
        node_x0 = tokens[:, :n_max]
        edge_x0 = tokens[:, n_max:]
        node_mask = mask_indicators[:, :n_max] & pad_mask[:, :n_max]
        edge_mask = mask_indicators[:, n_max:] & pad_mask[:, n_max:]

        alpha_64 = alpha_per_pos.double()
        alpha_prime_64 = alpha_prime_per_pos.double()
        denom = 1.0 - alpha_64 + 1e-8
        w_all = (-alpha_prime_64 / denom).float()
        w_all = torch.clamp(w_all, max=1000.0)
        w_node = w_all[:, :n_max]
        w_edge = w_all[:, n_max:]

        safe_node = torch.where(node_mask, node_x0, torch.zeros_like(node_x0))
        safe_edge = torch.where(edge_mask, edge_x0, torch.zeros_like(edge_x0))

        node_ce = F.cross_entropy(
            node_logits.reshape(-1, NODE_VOCAB_SIZE),
            safe_node.reshape(-1),
            reduction="none",
        ).reshape(B, -1)
        edge_ce = F.cross_entropy(
            edge_logits.reshape(-1, EDGE_VOCAB_SIZE),
            safe_edge.reshape(-1),
            reduction="none",
        ).reshape(B, -1)

        node_ce_masked = (node_ce * node_mask.float() * w_node).sum(dim=1)
        edge_ce_masked = (edge_ce * edge_mask.float() * w_edge).sum(dim=1)

        # Separate normalization (what ELBOLossV2 should do)
        expected_separate = (
            node_ce_masked / max(n_active_nodes, 1)
            + edge_ce_masked / max(n_active_edges, 1)
        ).mean()

        # Combined normalization (what we do NOT want)
        expected_combined = (
            (node_ce_masked + edge_ce_masked) / max(n_active_combined, 1)
        ).mean()

        # The actual loss should match separate, not combined
        assert torch.isclose(loss, expected_separate, atol=1e-4), (
            f"Loss {loss.item()} != expected separate {expected_separate.item()}"
        )
        # And should NOT match combined (unless edge_ce == 0, which is unlikely)
        assert not torch.isclose(loss, expected_combined, atol=1e-4), (
            f"Loss unexpectedly matches combined normalization: {loss.item()}"
        )
