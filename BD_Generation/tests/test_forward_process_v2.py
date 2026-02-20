"""Tests for v2 learned forward process: STGS sampling and learned masking.

Covers:
- STGS shape, normalization, hard-in-forward, gradient flow (spec 14.2: 11-14)
- Learned forward masking: soft emb shape, x_t dtype, mask_indicators bool,
  PAD never masked, boundary behavior at t~0 and t~1 (spec 14.3: 17-23)
- Eval discrete path returns same signature as v1 (spec 14.3: 23)
"""

from __future__ import annotations

import pytest
import torch

from bd_gen.data.vocab import (
    EDGE_MASK_IDX,
    EDGE_PAD_IDX,
    NODE_MASK_IDX,
    NODE_PAD_IDX,
    RPLAN_VOCAB_CONFIG,
    VocabConfig,
)
from bd_gen.diffusion.forward_process import (
    forward_mask_eval_learned,
    forward_mask_learned,
    stgs_sample,
)
from bd_gen.diffusion.rate_network import RateNetwork
from bd_gen.model.denoiser import BDDenoiser


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def vc() -> VocabConfig:
    """RPLAN VocabConfig."""
    return RPLAN_VOCAB_CONFIG


@pytest.fixture
def d_model() -> int:
    return 32


@pytest.fixture
def rate_net(vc: VocabConfig) -> RateNetwork:
    """Small RateNetwork for testing."""
    return RateNetwork(vocab_config=vc, d_emb=16, K=3, hidden_dim=32)


@pytest.fixture
def denoiser(vc: VocabConfig, d_model: int) -> BDDenoiser:
    """Small BDDenoiser for testing."""
    return BDDenoiser(
        d_model=d_model,
        n_layers=1,
        n_heads=2,
        vocab_config=vc,
        dropout=0.0,
    )


@pytest.fixture
def sample_data(vc: VocabConfig):
    """Create a batch of 4 samples with varying num_rooms.

    Returns dict with tokens, pad_mask, and timestep tensors.
    """
    num_rooms_list = [2, 4, 6, 8]
    B = len(num_rooms_list)
    tokens = torch.zeros(B, vc.seq_len, dtype=torch.long)
    pad_mask = torch.zeros(B, vc.seq_len, dtype=torch.bool)

    torch.manual_seed(123)

    for b, nr in enumerate(num_rooms_list):
        mask = vc.compute_pad_mask(nr)
        pad_mask[b] = mask
        # Fill node positions
        for k in range(vc.n_max):
            if k < nr:
                tokens[b, k] = torch.randint(0, 13, (1,))
            else:
                tokens[b, k] = NODE_PAD_IDX
        # Fill edge positions
        for pos in range(vc.n_edges):
            seq_idx = vc.n_max + pos
            if mask[seq_idx]:
                tokens[b, seq_idx] = torch.randint(0, 11, (1,))
            else:
                tokens[b, seq_idx] = EDGE_PAD_IDX

    t = torch.tensor([0.2, 0.4, 0.6, 0.8])
    return {"tokens": tokens, "pad_mask": pad_mask, "t": t}


# =========================================================================
# STGS Tests (spec 14.2: tests 11-14)
# =========================================================================


class TestSTGSSample:
    """Tests for stgs_sample function."""

    def test_shape(self, vc: VocabConfig):
        """Test 11: Output shape is (B, SEQ_LEN, 2)."""
        B = 4
        alpha = torch.rand(B, vc.seq_len) * 0.8 + 0.1  # in (0.1, 0.9)
        result = stgs_sample(alpha)
        assert result.shape == (B, vc.seq_len, 2)

    def test_sum_to_one(self, vc: VocabConfig):
        """Test 12: Weights sum to 1 along last dimension."""
        B = 4
        alpha = torch.rand(B, vc.seq_len) * 0.8 + 0.1
        result = stgs_sample(alpha)
        sums = result.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_hard_in_forward(self, vc: VocabConfig):
        """Test 13: Each position has exactly one 1 and one 0 in forward."""
        B = 8
        alpha = torch.rand(B, vc.seq_len) * 0.8 + 0.1
        result = stgs_sample(alpha)

        # Each row of the last dim should be either [1, 0] or [0, 1]
        # Check that max is 1.0 and min is 0.0 per position
        assert torch.allclose(
            result.max(dim=-1).values,
            torch.ones(B, vc.seq_len),
            atol=1e-5,
        )
        assert torch.allclose(
            result.min(dim=-1).values,
            torch.zeros(B, vc.seq_len),
            atol=1e-5,
        )

    def test_gradient_flow(self, vc: VocabConfig):
        """Test 14: Gradients flow back through STGS to alpha."""
        B = 2
        alpha = torch.rand(B, vc.seq_len) * 0.8 + 0.1
        alpha.requires_grad_(True)

        result = stgs_sample(alpha)
        loss = result.sum()
        loss.backward()

        assert alpha.grad is not None
        # Gradient should be non-zero for at least some positions
        assert alpha.grad.abs().sum() > 0


# =========================================================================
# forward_mask_learned Tests (spec 14.3: tests 17-22)
# =========================================================================


class TestForwardMaskLearned:
    """Tests for forward_mask_learned (training-time v2 forward process)."""

    def test_soft_emb_shape(
        self,
        sample_data: dict,
        rate_net: RateNetwork,
        denoiser: BDDenoiser,
        vc: VocabConfig,
        d_model: int,
    ):
        """Test 17: soft_embeddings.shape == (B, SEQ_LEN, d_model)."""
        output = forward_mask_learned(
            x0=sample_data["tokens"],
            pad_mask=sample_data["pad_mask"],
            t=sample_data["t"],
            rate_network=rate_net,
            denoiser=denoiser,
            vocab_config=vc,
        )
        B = sample_data["tokens"].shape[0]
        assert output["soft_embeddings"].shape == (B, vc.seq_len, d_model)

    def test_x_t_shape_and_dtype(
        self,
        sample_data: dict,
        rate_net: RateNetwork,
        denoiser: BDDenoiser,
        vc: VocabConfig,
    ):
        """Test 18: x_t.shape == (B, SEQ_LEN) and dtype is long."""
        output = forward_mask_learned(
            x0=sample_data["tokens"],
            pad_mask=sample_data["pad_mask"],
            t=sample_data["t"],
            rate_network=rate_net,
            denoiser=denoiser,
            vocab_config=vc,
        )
        B = sample_data["tokens"].shape[0]
        assert output["x_t"].shape == (B, vc.seq_len)
        assert output["x_t"].dtype == torch.long

    def test_mask_indicators_bool(
        self,
        sample_data: dict,
        rate_net: RateNetwork,
        denoiser: BDDenoiser,
        vc: VocabConfig,
    ):
        """Test 19: mask_indicators dtype is bool."""
        output = forward_mask_learned(
            x0=sample_data["tokens"],
            pad_mask=sample_data["pad_mask"],
            t=sample_data["t"],
            rate_network=rate_net,
            denoiser=denoiser,
            vocab_config=vc,
        )
        assert output["mask_indicators"].dtype == torch.bool

    def test_pad_never_masked(
        self,
        sample_data: dict,
        rate_net: RateNetwork,
        denoiser: BDDenoiser,
        vc: VocabConfig,
    ):
        """Test 20: PAD positions are never masked (mask_indicators[~pad_mask] == False)."""
        pad_mask = sample_data["pad_mask"]
        output = forward_mask_learned(
            x0=sample_data["tokens"],
            pad_mask=pad_mask,
            t=sample_data["t"],
            rate_network=rate_net,
            denoiser=denoiser,
            vocab_config=vc,
        )
        # No PAD position should be marked as masked
        assert not output["mask_indicators"][~pad_mask].any(), (
            "PAD position was masked in forward_mask_learned!"
        )

    def test_t_near_zero_almost_clean(
        self,
        vc: VocabConfig,
        rate_net: RateNetwork,
        denoiser: BDDenoiser,
    ):
        """Test 21: At t~0, almost no positions are masked (<5% of active)."""
        B = 16
        # Use max rooms (all positions active) for cleaner statistics
        pad_mask = vc.compute_pad_mask(vc.n_max).unsqueeze(0).expand(B, -1)

        torch.manual_seed(99)
        tokens = torch.zeros(B, vc.seq_len, dtype=torch.long)
        tokens[:, :vc.n_max] = torch.randint(0, 13, (B, vc.n_max))
        tokens[:, vc.n_max:] = torch.randint(0, 11, (B, vc.n_edges))

        t = torch.full((B,), 1e-5)
        output = forward_mask_learned(
            x0=tokens,
            pad_mask=pad_mask,
            t=t,
            rate_network=rate_net,
            denoiser=denoiser,
            vocab_config=vc,
        )
        mask_count = output["mask_indicators"].sum().item()
        active_count = pad_mask.sum().item()
        assert mask_count < 0.05 * active_count, (
            f"At t~0, expected <5% masked, got {mask_count}/{active_count} "
            f"= {mask_count / active_count:.2%}"
        )

    def test_t_near_one_almost_masked(
        self,
        vc: VocabConfig,
        rate_net: RateNetwork,
        denoiser: BDDenoiser,
    ):
        """Test 22: At t~1, almost all positions are masked (>95% of active)."""
        B = 16
        pad_mask = vc.compute_pad_mask(vc.n_max).unsqueeze(0).expand(B, -1)

        torch.manual_seed(99)
        tokens = torch.zeros(B, vc.seq_len, dtype=torch.long)
        tokens[:, :vc.n_max] = torch.randint(0, 13, (B, vc.n_max))
        tokens[:, vc.n_max:] = torch.randint(0, 11, (B, vc.n_edges))

        t = torch.full((B,), 1.0 - 1e-5)
        output = forward_mask_learned(
            x0=tokens,
            pad_mask=pad_mask,
            t=t,
            rate_network=rate_net,
            denoiser=denoiser,
            vocab_config=vc,
        )
        mask_count = output["mask_indicators"].sum().item()
        active_count = pad_mask.sum().item()
        assert mask_count > 0.95 * active_count, (
            f"At t~1, expected >95% masked, got {mask_count}/{active_count} "
            f"= {mask_count / active_count:.2%}"
        )


# =========================================================================
# forward_mask_eval_learned Tests (spec 14.3: test 23)
# =========================================================================


class TestForwardMaskEvalLearned:
    """Tests for forward_mask_eval_learned (eval-time v2 forward process)."""

    def test_eval_discrete_path_signature(
        self,
        sample_data: dict,
        rate_net: RateNetwork,
        vc: VocabConfig,
    ):
        """Test 23: Returns same (x_t, mask_indicators) signature as v1 forward_mask."""
        result = forward_mask_eval_learned(
            x0=sample_data["tokens"],
            pad_mask=sample_data["pad_mask"],
            t=sample_data["t"],
            rate_network=rate_net,
            vocab_config=vc,
        )
        # Should return a tuple of two tensors
        assert isinstance(result, tuple)
        assert len(result) == 2

        x_t, mask_indicators = result
        B = sample_data["tokens"].shape[0]

        # x_t shape and dtype
        assert x_t.shape == (B, vc.seq_len)
        assert x_t.dtype == torch.long

        # mask_indicators shape and dtype
        assert mask_indicators.shape == (B, vc.seq_len)
        assert mask_indicators.dtype == torch.bool

    def test_eval_pad_never_masked(
        self,
        sample_data: dict,
        rate_net: RateNetwork,
        vc: VocabConfig,
    ):
        """PAD positions must never be masked in eval path."""
        pad_mask = sample_data["pad_mask"]
        x_t, mask_indicators = forward_mask_eval_learned(
            x0=sample_data["tokens"],
            pad_mask=pad_mask,
            t=sample_data["t"],
            rate_network=rate_net,
            vocab_config=vc,
        )
        assert not mask_indicators[~pad_mask].any(), (
            "PAD position was masked in forward_mask_eval_learned!"
        )

    def test_eval_correct_mask_tokens(
        self,
        vc: VocabConfig,
        rate_net: RateNetwork,
    ):
        """Masked nodes get NODE_MASK_IDX, masked edges get EDGE_MASK_IDX."""
        B = 8
        pad_mask = vc.compute_pad_mask(vc.n_max).unsqueeze(0).expand(B, -1)

        torch.manual_seed(77)
        tokens = torch.zeros(B, vc.seq_len, dtype=torch.long)
        tokens[:, :vc.n_max] = torch.randint(0, 13, (B, vc.n_max))
        tokens[:, vc.n_max:] = torch.randint(0, 11, (B, vc.n_edges))

        t = torch.full((B,), 0.5)
        x_t, mask_indicators = forward_mask_eval_learned(
            x0=tokens, pad_mask=pad_mask, t=t,
            rate_network=rate_net, vocab_config=vc,
        )

        n_max = vc.n_max
        # Masked node positions should have NODE_MASK_IDX
        node_masked = mask_indicators[:, :n_max]
        if node_masked.any():
            assert (x_t[:, :n_max][node_masked] == NODE_MASK_IDX).all()

        # Masked edge positions should have EDGE_MASK_IDX
        edge_masked = mask_indicators[:, n_max:]
        if edge_masked.any():
            assert (x_t[:, n_max:][edge_masked] == EDGE_MASK_IDX).all()
