"""Tests for v2 learned forward process support in reverse sampling.

Covers spec Section 14.6 tests:
  35. No MASK tokens in generated samples (real positions)
  36. PAD positions contain correct PAD tokens
  37. Output shape is (B, SEQ_LEN) long
  38. rate_network=None behaves identically to v1 (backward compatible)

Also tests v2 + remasking integration (rate_network + remasking_fn).
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
)
from bd_gen.diffusion.noise_schedule import LogLinearSchedule
from bd_gen.diffusion.rate_network import RateNetwork
from bd_gen.diffusion.sampling import sample
from bd_gen.model.denoiser import BDDenoiser

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

vc = RPLAN_VOCAB_CONFIG


@pytest.fixture
def small_model():
    """Small BDDenoiser for fast test execution."""
    return BDDenoiser(
        d_model=32,
        n_layers=1,
        n_heads=2,
        vocab_config=vc,
        dropout=0.0,
    )


@pytest.fixture
def rate_net():
    """RateNetwork for v2 per-position alpha."""
    return RateNetwork(vc)


@pytest.fixture
def loglinear():
    """LogLinearSchedule (still required as positional arg)."""
    return LogLinearSchedule()


# ---------------------------------------------------------------------------
# Test 35: No MASK tokens in real positions
# ---------------------------------------------------------------------------


class TestV2NoMaskTokens:
    """Generated samples have no NODE_MASK_IDX or EDGE_MASK_IDX in real positions."""

    def test_random_mode_no_mask(self, small_model, loglinear, rate_net):
        """random unmasking with rate_network: no MASK in output."""
        result = sample(
            small_model, loglinear, vc,
            batch_size=4, num_steps=10,
            unmasking_mode="random",
            rate_network=rate_net,
            fixed_num_rooms=4,
        )
        # Check node positions (first n_max)
        assert (result[:, :vc.n_max] != NODE_MASK_IDX).all(), (
            "Found NODE_MASK_IDX in node positions"
        )
        # Check edge positions (after n_max)
        pad_mask = torch.stack(
            [vc.compute_pad_mask(4) for _ in range(4)]
        )
        edge_real = pad_mask[:, vc.n_max:]
        edge_tokens = result[:, vc.n_max:]
        assert (edge_tokens[edge_real] != EDGE_MASK_IDX).all(), (
            "Found EDGE_MASK_IDX in real edge positions"
        )

    def test_llada_mode_no_mask(self, small_model, loglinear, rate_net):
        """llada unmasking with rate_network: no MASK in output."""
        result = sample(
            small_model, loglinear, vc,
            batch_size=4, num_steps=10,
            unmasking_mode="llada",
            rate_network=rate_net,
            fixed_num_rooms=4,
        )
        assert (result[:, :vc.n_max] != NODE_MASK_IDX).all()
        pad_mask = torch.stack(
            [vc.compute_pad_mask(4) for _ in range(4)]
        )
        edge_real = pad_mask[:, vc.n_max:]
        edge_tokens = result[:, vc.n_max:]
        assert (edge_tokens[edge_real] != EDGE_MASK_IDX).all()

    def test_max_rooms_no_mask(self, small_model, loglinear, rate_net):
        """All positions real (8 rooms): no MASK anywhere."""
        result = sample(
            small_model, loglinear, vc,
            batch_size=2, num_steps=10,
            rate_network=rate_net,
            fixed_num_rooms=8,
        )
        assert (result[:, :vc.n_max] != NODE_MASK_IDX).all()
        assert (result[:, vc.n_max:] != EDGE_MASK_IDX).all()


# ---------------------------------------------------------------------------
# Test 36: PAD positions contain correct PAD tokens
# ---------------------------------------------------------------------------


class TestV2PadCorrect:
    """PAD positions contain NODE_PAD_IDX / EDGE_PAD_IDX."""

    def test_random_mode_pad(self, small_model, loglinear, rate_net):
        num_rooms = 3
        result = sample(
            small_model, loglinear, vc,
            batch_size=4, num_steps=10,
            unmasking_mode="random",
            rate_network=rate_net,
            fixed_num_rooms=num_rooms,
        )
        n_max = vc.n_max
        # Node PAD positions
        assert (result[:, num_rooms:n_max] == NODE_PAD_IDX).all(), (
            "Node PAD positions not NODE_PAD_IDX"
        )
        # Edge PAD positions
        pad_mask = vc.compute_pad_mask(num_rooms)
        for pos in range(vc.n_edges):
            if not pad_mask[n_max + pos]:
                assert (result[:, n_max + pos] == EDGE_PAD_IDX).all(), (
                    f"Edge PAD position {pos} not EDGE_PAD_IDX"
                )

    def test_llada_mode_pad(self, small_model, loglinear, rate_net):
        num_rooms = 3
        result = sample(
            small_model, loglinear, vc,
            batch_size=4, num_steps=10,
            unmasking_mode="llada",
            rate_network=rate_net,
            fixed_num_rooms=num_rooms,
        )
        n_max = vc.n_max
        assert (result[:, num_rooms:n_max] == NODE_PAD_IDX).all()
        pad_mask = vc.compute_pad_mask(num_rooms)
        for pos in range(vc.n_edges):
            if not pad_mask[n_max + pos]:
                assert (result[:, n_max + pos] == EDGE_PAD_IDX).all()


# ---------------------------------------------------------------------------
# Test 37: Output shape (B, SEQ_LEN) long
# ---------------------------------------------------------------------------


class TestV2OutputShape:
    """Output is (B, SEQ_LEN) long tensor."""

    def test_random_mode_shape(self, small_model, loglinear, rate_net):
        result = sample(
            small_model, loglinear, vc,
            batch_size=3, num_steps=10,
            unmasking_mode="random",
            rate_network=rate_net,
            fixed_num_rooms=4,
        )
        assert result.shape == (3, vc.seq_len)
        assert result.dtype == torch.long

    def test_llada_mode_shape(self, small_model, loglinear, rate_net):
        result = sample(
            small_model, loglinear, vc,
            batch_size=3, num_steps=10,
            unmasking_mode="llada",
            rate_network=rate_net,
            fixed_num_rooms=4,
        )
        assert result.shape == (3, vc.seq_len)
        assert result.dtype == torch.long


# ---------------------------------------------------------------------------
# Test 38: rate_network=None backward compatible with v1
# ---------------------------------------------------------------------------


class TestV2BackwardCompat:
    """rate_network=None behaves identically to v1."""

    def test_none_matches_v1_random(self, small_model, loglinear):
        """Explicit rate_network=None gives same output as omitting it."""
        torch.manual_seed(42)
        r_v1 = sample(
            small_model, loglinear, vc,
            batch_size=2, num_steps=5,
            fixed_num_rooms=4,
        )
        torch.manual_seed(42)
        r_v2_none = sample(
            small_model, loglinear, vc,
            batch_size=2, num_steps=5,
            rate_network=None,
            fixed_num_rooms=4,
        )
        assert torch.equal(r_v1, r_v2_none), (
            "rate_network=None should be identical to v1 (no rate_network arg)"
        )

    def test_none_matches_v1_llada(self, small_model, loglinear):
        """Explicit rate_network=None gives same output as omitting it (llada mode)."""
        torch.manual_seed(42)
        r_v1 = sample(
            small_model, loglinear, vc,
            batch_size=2, num_steps=5,
            unmasking_mode="llada",
            fixed_num_rooms=4,
        )
        torch.manual_seed(42)
        r_v2_none = sample(
            small_model, loglinear, vc,
            batch_size=2, num_steps=5,
            unmasking_mode="llada",
            rate_network=None,
            fixed_num_rooms=4,
        )
        assert torch.equal(r_v1, r_v2_none)


# ---------------------------------------------------------------------------
# Remasking warning
# ---------------------------------------------------------------------------


class TestV2RemaskingIntegration:
    """Remasking works correctly with v2 rate_network."""

    def test_remasking_called_with_rate_network(self, small_model, loglinear, rate_net):
        """Providing both rate_network and remasking_fn: remasking IS called."""
        call_count = [0]

        def tracking_remasking(x_t, t_now, t_next, pad_mask, **kwargs):
            call_count[0] += 1
            return x_t

        result = sample(
            small_model, loglinear, vc,
            batch_size=2, num_steps=5,
            rate_network=rate_net,
            remasking_fn=tracking_remasking,
            fixed_num_rooms=4,
        )

        # Remasking called at steps where i > 0 AND t_now < t_switch (1.0).
        # With 5 steps: i=4 (t=1.0, skipped: not < 1.0), i=3,2,1 (called),
        # i=0 (skipped: last step). So 3 calls.
        assert call_count[0] == 3, (
            f"remasking_fn should be called 3 times "
            f"but was called {call_count[0]}"
        )

        # Output still valid
        assert result.shape == (2, vc.seq_len)
        assert result.dtype == torch.long

    def test_confidence_remasking_with_v2_no_mask(
        self, small_model, loglinear, rate_net,
    ):
        """v2 + confidence remasking: no MASK tokens in output."""
        from bd_gen.diffusion.remasking import RemaskingSchedule

        remasking_fn = RemaskingSchedule(
            strategy="confidence",
            eta=0.0,
            noise_schedule=None,
            vocab_config=vc,
            rate_network=rate_net,
        )

        result = sample(
            small_model, loglinear, vc,
            batch_size=4, num_steps=10,
            unmasking_mode="llada",
            top_p=0.9,
            rate_network=rate_net,
            remasking_fn=remasking_fn,
            fixed_num_rooms=4,
        )

        assert result.shape == (4, vc.seq_len)
        # No MASK tokens in real positions
        pad_mask = torch.stack([vc.compute_pad_mask(4) for _ in range(4)])
        for b in range(4):
            for pos in range(vc.seq_len):
                if pad_mask[b, pos]:
                    tok = result[b, pos].item()
                    if pos < vc.n_max:
                        assert tok != NODE_MASK_IDX, f"MASK at node pos {pos}"
                    else:
                        assert tok != EDGE_MASK_IDX, f"MASK at edge pos {pos}"

    def test_v2_remasking_pad_protected(self, small_model, loglinear, rate_net):
        """v2 + remasking: PAD positions remain correct."""
        from bd_gen.diffusion.remasking import RemaskingSchedule

        remasking_fn = RemaskingSchedule(
            strategy="confidence",
            eta=0.0,
            noise_schedule=None,
            vocab_config=vc,
            rate_network=rate_net,
        )

        num_rooms = 3
        result = sample(
            small_model, loglinear, vc,
            batch_size=4, num_steps=10,
            unmasking_mode="llada",
            top_p=0.9,
            rate_network=rate_net,
            remasking_fn=remasking_fn,
            fixed_num_rooms=num_rooms,
        )

        n_max = vc.n_max
        # Node PAD
        assert (result[:, num_rooms:n_max] == NODE_PAD_IDX).all()
        # Edge PAD
        pad_mask = vc.compute_pad_mask(num_rooms)
        for pos in range(vc.n_edges):
            if not pad_mask[n_max + pos]:
                assert (result[:, n_max + pos] == EDGE_PAD_IDX).all()
