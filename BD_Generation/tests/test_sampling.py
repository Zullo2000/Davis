"""Tests for reverse sampling (denoising).

Covers:
- Output shapes and dtypes
- No MASK tokens in final output
- PAD preservation throughout
- Temperature behavior (deterministic vs stochastic)
- num_rooms handling (fixed, distribution, uniform)
- Guidance function and inpainting hooks
- Adversarial sampling cases
"""

from __future__ import annotations

import torch

from bd_gen.data.vocab import (
    EDGE_MASK_IDX,
    EDGE_PAD_IDX,
    NODE_MASK_IDX,
    NODE_PAD_IDX,
)
from bd_gen.diffusion.sampling import sample

# =========================================================================
# Basic Correctness
# =========================================================================


class TestSamplingBasic:
    """Core correctness for sample()."""

    def test_output_shape(self, dummy_model, linear_schedule, vocab_config):
        result = sample(
            dummy_model, linear_schedule, vocab_config,
            batch_size=4, num_steps=5, fixed_num_rooms=4,
        )
        assert result.shape == (4, vocab_config.seq_len)

    def test_output_dtype(self, dummy_model, linear_schedule, vocab_config):
        result = sample(
            dummy_model, linear_schedule, vocab_config,
            batch_size=2, num_steps=5, fixed_num_rooms=4,
        )
        assert result.dtype == torch.long

    def test_no_mask_tokens_in_output(
        self, dummy_model, linear_schedule, vocab_config
    ):
        """No NODE_MASK_IDX or EDGE_MASK_IDX in non-PAD positions."""
        result = sample(
            dummy_model, linear_schedule, vocab_config,
            batch_size=4, num_steps=10, fixed_num_rooms=8,
        )
        # For num_rooms=8, all positions are real (no PAD)
        assert (result[:, : vocab_config.n_max] != NODE_MASK_IDX).all()
        assert (result[:, vocab_config.n_max :] != EDGE_MASK_IDX).all()

    def test_pad_positions_preserved(
        self, dummy_model, linear_schedule, vocab_config
    ):
        """PAD positions have correct PAD token values."""
        result = sample(
            dummy_model, linear_schedule, vocab_config,
            batch_size=4, num_steps=5, fixed_num_rooms=3,
        )
        n_max = vocab_config.n_max
        # Node PAD positions (indices 3-7 for num_rooms=3)
        assert (result[:, 3:n_max] == NODE_PAD_IDX).all()
        # Edge PAD positions (where either endpoint >= 3)
        pad_mask = vocab_config.compute_pad_mask(3)
        edge_pad = ~pad_mask[n_max:]
        for pos in range(vocab_config.n_edges):
            if edge_pad[pos]:
                assert (result[:, n_max + pos] == EDGE_PAD_IDX).all()

    def test_node_tokens_valid_range(
        self, dummy_model, linear_schedule, vocab_config
    ):
        """Non-PAD node tokens are in [0, 14] (valid node vocab indices)."""
        result = sample(
            dummy_model, linear_schedule, vocab_config,
            batch_size=4, num_steps=10, fixed_num_rooms=8,
        )
        node_tokens = result[:, : vocab_config.n_max]
        assert (node_tokens >= 0).all()
        assert (node_tokens < 15).all()

    def test_edge_tokens_valid_range(
        self, dummy_model, linear_schedule, vocab_config
    ):
        """Non-PAD edge tokens are in [0, 12] (valid edge vocab indices)."""
        result = sample(
            dummy_model, linear_schedule, vocab_config,
            batch_size=4, num_steps=10, fixed_num_rooms=8,
        )
        edge_tokens = result[:, vocab_config.n_max :]
        assert (edge_tokens >= 0).all()
        assert (edge_tokens < 13).all()


# =========================================================================
# Temperature
# =========================================================================


class TestTemperature:
    """Verify temperature behavior."""

    def test_temperature_zero_deterministic(
        self, dummy_model, linear_schedule, vocab_config
    ):
        """Two calls with temperature=0 and same seed -> identical output."""
        torch.manual_seed(42)
        r1 = sample(
            dummy_model, linear_schedule, vocab_config,
            batch_size=2, num_steps=5, temperature=0.0, fixed_num_rooms=4,
        )
        torch.manual_seed(42)
        r2 = sample(
            dummy_model, linear_schedule, vocab_config,
            batch_size=2, num_steps=5, temperature=0.0, fixed_num_rooms=4,
        )
        assert torch.equal(r1, r2)

    def test_temperature_positive_stochastic(
        self, dummy_model, linear_schedule, vocab_config
    ):
        """Two calls with temp>0, different seeds -> likely different output."""
        torch.manual_seed(1)
        r1 = sample(
            dummy_model, linear_schedule, vocab_config,
            batch_size=4, num_steps=10, temperature=1.0, fixed_num_rooms=8,
        )
        torch.manual_seed(2)
        r2 = sample(
            dummy_model, linear_schedule, vocab_config,
            batch_size=4, num_steps=10, temperature=1.0, fixed_num_rooms=8,
        )
        # Very unlikely to be identical with temp=1.0
        assert not torch.equal(r1, r2)

    def test_degenerate_gumbel_matches_argmax(self, vocab_config):
        """At very low temperature, Gumbel argmax == raw argmax for
        logits with clear peaks (large gap between max and runner-up).

        Uses synthetic logits, not the zero-init model (which has near-
        uniform logits where Gumbel noise dominates at any temperature).
        """
        from bd_gen.diffusion.sampling import _gumbel_sample

        B, n_max, n_edges = 4, vocab_config.n_max, vocab_config.n_edges

        # Create logits with clear peaks (gap >> Gumbel noise scale)
        node_logits = torch.randn(B, n_max, 15) * 0.1
        edge_logits = torch.randn(B, n_edges, 13) * 0.1
        # Set one class to have a large logit value per position
        for b in range(B):
            for k in range(n_max):
                node_logits[b, k, (b + k) % 15] = 5.0
            for k in range(n_edges):
                edge_logits[b, k, (b + k) % 13] = 5.0

        node_argmax = node_logits.argmax(dim=-1)
        edge_argmax = edge_logits.argmax(dim=-1)

        # Gumbel with very low temperature: logits/1e-10 ~ 5e10 >> noise
        node_gumbel = _gumbel_sample(node_logits, 1e-10, node_logits.device)
        edge_gumbel = _gumbel_sample(edge_logits, 1e-10, edge_logits.device)

        assert torch.equal(node_argmax, node_gumbel.argmax(dim=-1))
        assert torch.equal(edge_argmax, edge_gumbel.argmax(dim=-1))


# =========================================================================
# num_rooms Handling
# =========================================================================


class TestNumRooms:
    """Verify num_rooms handling."""

    def test_fixed_num_rooms(self, dummy_model, linear_schedule, vocab_config):
        """All samples have exactly fixed_num_rooms real node positions."""
        result = sample(
            dummy_model, linear_schedule, vocab_config,
            batch_size=4, num_steps=5, fixed_num_rooms=5,
        )
        n_max = vocab_config.n_max
        for b in range(4):
            # First 5 nodes should not be PAD
            assert (result[b, :5] != NODE_PAD_IDX).all()
            # Remaining nodes should be PAD
            assert (result[b, 5:n_max] == NODE_PAD_IDX).all()

    def test_distribution_sampling(
        self, dummy_model, linear_schedule, vocab_config
    ):
        """With a distribution, room counts should be valid."""
        # Distribution: only 4-room graphs
        dist = torch.zeros(vocab_config.n_max)
        dist[3] = 1.0  # index 3 = 4 rooms

        result = sample(
            dummy_model, linear_schedule, vocab_config,
            batch_size=4, num_steps=5,
            num_rooms_distribution=dist,
        )
        n_max = vocab_config.n_max
        for b in range(4):
            assert (result[b, :4] != NODE_PAD_IDX).all()
            assert (result[b, 4:n_max] == NODE_PAD_IDX).all()

    def test_uniform_fallback(self, dummy_model, linear_schedule, vocab_config):
        """When no distribution or fixed provided, still produces valid output."""
        torch.manual_seed(42)
        result = sample(
            dummy_model, linear_schedule, vocab_config,
            batch_size=10, num_steps=5,
        )
        assert result.shape == (10, vocab_config.seq_len)
        # Should have various room counts
        n_max = vocab_config.n_max
        room_counts = []
        for b in range(10):
            nr = (result[b, :n_max] != NODE_PAD_IDX).sum().item()
            room_counts.append(nr)
            assert 1 <= nr <= n_max


# =========================================================================
# PAD Preservation
# =========================================================================


class TestPadPreservation:
    """PAD positions must remain PAD throughout."""

    def test_pad_preserved_small_steps(
        self, dummy_model, linear_schedule, vocab_config
    ):
        result = sample(
            dummy_model, linear_schedule, vocab_config,
            batch_size=4, num_steps=5, fixed_num_rooms=3,
        )
        pad_mask = vocab_config.compute_pad_mask(3)
        n_max = vocab_config.n_max
        for b in range(4):
            # Node PAD
            for k in range(3, n_max):
                assert result[b, k].item() == NODE_PAD_IDX
            # Edge PAD
            for pos in range(vocab_config.n_edges):
                if not pad_mask[n_max + pos]:
                    assert result[b, n_max + pos].item() == EDGE_PAD_IDX

    def test_pad_preserved_single_step(
        self, dummy_model, linear_schedule, vocab_config
    ):
        result = sample(
            dummy_model, linear_schedule, vocab_config,
            batch_size=2, num_steps=1, fixed_num_rooms=2,
        )
        n_max = vocab_config.n_max
        assert (result[:, 2:n_max] == NODE_PAD_IDX).all()

    def test_pad_preserved_many_steps(
        self, dummy_model, linear_schedule, vocab_config
    ):
        result = sample(
            dummy_model, linear_schedule, vocab_config,
            batch_size=2, num_steps=50, fixed_num_rooms=4,
        )
        n_max = vocab_config.n_max
        assert (result[:, 4:n_max] == NODE_PAD_IDX).all()


# =========================================================================
# Guidance and Inpainting Hooks
# =========================================================================


class TestGuidanceAndInpainting:
    """Verify pluggable hooks work correctly."""

    def test_guidance_fn_called(
        self, dummy_model, linear_schedule, vocab_config
    ):
        """A guidance_fn that tracks calls is invoked num_steps times."""
        call_count = [0]

        def tracking_guidance(logits_tuple, x_t, t, pad_mask):
            call_count[0] += 1
            return logits_tuple

        num_steps = 7
        sample(
            dummy_model, linear_schedule, vocab_config,
            batch_size=2, num_steps=num_steps, fixed_num_rooms=4,
            guidance_fn=tracking_guidance,
        )
        assert call_count[0] == num_steps

    def test_fixed_tokens_preserved(
        self, dummy_model, linear_schedule, vocab_config
    ):
        """Positions marked in fixed_mask have fixed_tokens values in output."""
        seq_len = vocab_config.seq_len
        B = 2

        fixed_tokens = torch.zeros(B, seq_len, dtype=torch.long)
        fixed_mask = torch.zeros(B, seq_len, dtype=torch.bool)

        # Fix the first two node positions to specific values
        fixed_tokens[:, 0] = 3  # Bathroom
        fixed_tokens[:, 1] = 7  # SecondRoom
        fixed_mask[:, 0] = True
        fixed_mask[:, 1] = True

        result = sample(
            dummy_model, linear_schedule, vocab_config,
            batch_size=B, num_steps=10, fixed_num_rooms=4,
            fixed_tokens=fixed_tokens, fixed_mask=fixed_mask,
        )
        assert (result[:, 0] == 3).all()
        assert (result[:, 1] == 7).all()

    def test_remasking_fn_called(
        self, dummy_model, linear_schedule, vocab_config
    ):
        """A remasking_fn that tracks calls is invoked at each step."""
        call_count = [0]

        def tracking_remasking(x_t, t):
            call_count[0] += 1
            return x_t

        num_steps = 5
        sample(
            dummy_model, linear_schedule, vocab_config,
            batch_size=2, num_steps=num_steps, fixed_num_rooms=4,
            remasking_fn=tracking_remasking,
        )
        assert call_count[0] == num_steps


# =========================================================================
# Adversarial Sampling Cases (Spec Section 7.4)
# =========================================================================


class TestAdversarialSampling:
    """Adversarial sampling cases from spec Section 7.4."""

    def test_single_room_sampling(
        self, dummy_model, linear_schedule, vocab_config
    ):
        """fixed_num_rooms=1: output has exactly 1 non-PAD node."""
        result = sample(
            dummy_model, linear_schedule, vocab_config,
            batch_size=4, num_steps=10, fixed_num_rooms=1,
        )
        n_max = vocab_config.n_max
        for b in range(4):
            # Exactly 1 real node
            assert result[b, 0].item() != NODE_PAD_IDX
            assert (result[b, 1:n_max] == NODE_PAD_IDX).all()
            # All edges are PAD (1 room -> no edges)
            assert (result[b, n_max:] == EDGE_PAD_IDX).all()
            # No MASK tokens
            assert result[b, 0].item() != NODE_MASK_IDX

    def test_max_room_sampling(
        self, dummy_model, linear_schedule, vocab_config
    ):
        """fixed_num_rooms=8: output has 0 PAD positions."""
        result = sample(
            dummy_model, linear_schedule, vocab_config,
            batch_size=4, num_steps=10, fixed_num_rooms=8,
        )
        assert (result[:, : vocab_config.n_max] != NODE_PAD_IDX).all()
        assert (result[:, vocab_config.n_max :] != EDGE_PAD_IDX).all()
        assert (result[:, : vocab_config.n_max] != NODE_MASK_IDX).all()
        assert (result[:, vocab_config.n_max :] != EDGE_MASK_IDX).all()

    def test_final_cleanup_removes_all_mask(
        self, dummy_model, linear_schedule, vocab_config
    ):
        """Even with num_steps=1 (poor unmasking), final cleanup handles it."""
        result = sample(
            dummy_model, linear_schedule, vocab_config,
            batch_size=4, num_steps=1, fixed_num_rooms=8,
        )
        # No MASK tokens should remain
        assert (result[:, : vocab_config.n_max] != NODE_MASK_IDX).all()
        assert (result[:, vocab_config.n_max :] != EDGE_MASK_IDX).all()
