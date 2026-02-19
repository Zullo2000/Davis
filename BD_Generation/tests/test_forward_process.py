"""Tests for noise schedules and forward masking process.

Covers:
- Noise schedule math (linear and cosine)
- Factory function get_noise
- Forward masking with PAD protection
- PAD stress tests (10,000+ random operations)
- Statistical masking rate verification
"""

from __future__ import annotations

import math

import pytest
import torch
from omegaconf import OmegaConf

from bd_gen.data.vocab import (
    EDGE_MASK_IDX,
    EDGE_PAD_IDX,
    NODE_MASK_IDX,
    NODE_PAD_IDX,
    RPLAN_VOCAB_CONFIG,
)
from bd_gen.diffusion.forward_process import forward_mask
from bd_gen.diffusion.noise_schedule import (
    CosineSchedule,
    LinearSchedule,
    LogLinearSchedule,
    get_noise,
)

# =========================================================================
# Noise Schedule Tests
# =========================================================================


class TestLinearSchedule:
    """Tests for LinearSchedule mathematical correctness."""

    def test_sigma_at_t0(self, linear_schedule):
        t = torch.tensor(0.0)
        assert torch.isclose(linear_schedule.sigma(t), torch.tensor(0.0), atol=1e-7)

    def test_sigma_at_t1(self, linear_schedule):
        t = torch.tensor(1.0)
        assert torch.isclose(linear_schedule.sigma(t), torch.tensor(10.0), atol=1e-7)

    def test_alpha_at_t0(self, linear_schedule):
        t = torch.tensor(0.0)
        assert torch.isclose(linear_schedule.alpha(t), torch.tensor(1.0), atol=1e-6)

    def test_alpha_at_t1(self, linear_schedule):
        t = torch.tensor(1.0)
        expected = torch.exp(torch.tensor(-10.0))
        assert torch.isclose(linear_schedule.alpha(t), expected, atol=1e-7)

    def test_alpha_prime_sign(self, linear_schedule):
        """alpha_prime(t) < 0 for all t in (0, 1) since alpha is decreasing."""
        t = torch.linspace(0.01, 0.99, 100)
        ap = linear_schedule.alpha_prime(t)
        assert (ap < 0).all()

    def test_alpha_prime_formula(self, linear_schedule):
        """Verify alpha_prime matches -(sigma_max - sigma_min) * alpha(t)."""
        t = torch.linspace(0.0, 1.0, 50)
        expected = -10.0 * linear_schedule.alpha(t)
        actual = linear_schedule.alpha_prime(t)
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_alpha_monotonically_decreasing(self, linear_schedule):
        t = torch.linspace(0.0, 1.0, 200)
        alphas = linear_schedule.alpha(t)
        diffs = alphas[1:] - alphas[:-1]
        assert (diffs <= 0).all()

    def test_importance_sampling_in_range(self, linear_schedule):
        t = torch.linspace(0.0, 1.0, 100)
        transformed = linear_schedule.importance_sampling_transformation(t)
        assert (transformed >= -1e-6).all()
        assert (transformed <= 1.0 + 1e-6).all()

    def test_importance_sampling_monotonic(self, linear_schedule):
        t = torch.linspace(0.0, 1.0, 100)
        transformed = linear_schedule.importance_sampling_transformation(t)
        diffs = transformed[1:] - transformed[:-1]
        assert (diffs >= -1e-6).all()


class TestLogLinearSchedule:
    """Tests for LogLinearSchedule mathematical correctness."""

    def test_alpha_at_t0(self, loglinear_schedule):
        t = torch.tensor(0.0)
        expected = 1.0  # 1 - (1-eps)*0 = 1
        assert torch.isclose(loglinear_schedule.alpha(t), torch.tensor(expected), atol=1e-6)

    def test_alpha_at_t1(self, loglinear_schedule):
        t = torch.tensor(1.0)
        expected = 1e-3  # 1 - (1-eps)*1 = eps
        assert torch.isclose(loglinear_schedule.alpha(t), torch.tensor(expected), atol=1e-6)

    def test_alpha_at_t_half(self, loglinear_schedule):
        """Key property: alpha(0.5) ~ 0.5 (linear masking curve)."""
        t = torch.tensor(0.5)
        expected = 1.0 - (1.0 - 1e-3) * 0.5  # ~0.5005
        assert torch.isclose(loglinear_schedule.alpha(t), torch.tensor(expected), atol=1e-4)

    def test_alpha_prime_constant(self, loglinear_schedule):
        """alpha_prime should be constant = -(1-eps) for all t."""
        t = torch.linspace(0.0, 1.0, 50)
        ap = loglinear_schedule.alpha_prime(t)
        expected = -(1.0 - 1e-3)
        assert torch.allclose(ap, torch.full_like(ap, expected), atol=1e-6)

    def test_alpha_prime_sign(self, loglinear_schedule):
        """alpha_prime(t) < 0 for all t (alpha is decreasing)."""
        t = torch.linspace(0.01, 0.99, 100)
        ap = loglinear_schedule.alpha_prime(t)
        assert (ap < 0).all()

    def test_alpha_monotonically_decreasing(self, loglinear_schedule):
        t = torch.linspace(0.0, 1.0, 200)
        alphas = loglinear_schedule.alpha(t)
        diffs = alphas[1:] - alphas[:-1]
        assert (diffs <= 0).all()

    def test_sigma_nonnegative(self, loglinear_schedule):
        t = torch.linspace(0.0, 1.0, 200)
        sigmas = loglinear_schedule.sigma(t)
        assert (sigmas >= -1e-6).all()

    def test_sigma_consistent_with_alpha(self, loglinear_schedule):
        """exp(-sigma(t)) should equal alpha(t)."""
        t = torch.linspace(0.01, 0.99, 100)
        from_sigma = torch.exp(-loglinear_schedule.sigma(t))
        from_alpha = loglinear_schedule.alpha(t)
        assert torch.allclose(from_sigma, from_alpha, atol=1e-5)

    def test_importance_sampling_in_range(self, loglinear_schedule):
        t = torch.linspace(0.0, 1.0, 100)
        transformed = loglinear_schedule.importance_sampling_transformation(t)
        assert (transformed >= -1e-6).all()
        assert (transformed <= 1.0 + 1e-6).all()

    def test_importance_sampling_monotonic(self, loglinear_schedule):
        t = torch.linspace(0.0, 1.0, 100)
        transformed = loglinear_schedule.importance_sampling_transformation(t)
        diffs = transformed[1:] - transformed[:-1]
        assert (diffs >= -1e-6).all()


class TestCosineSchedule:
    """Tests for CosineSchedule mathematical correctness."""

    def test_alpha_at_t0(self, cosine_schedule):
        t = torch.tensor(0.0)
        expected = 1e-3 + (1 - 1e-3) * 1.0  # eps + (1-eps)*cos(0)
        actual = cosine_schedule.alpha(t)
        assert torch.isclose(actual, torch.tensor(expected), atol=1e-6)

    def test_alpha_at_t1(self, cosine_schedule):
        t = torch.tensor(1.0)
        expected = 1e-3  # eps + (1-eps)*cos(pi/2) = eps
        actual = cosine_schedule.alpha(t)
        assert torch.isclose(actual, torch.tensor(expected), atol=1e-6)

    def test_alpha_prime_at_t0(self, cosine_schedule):
        """alpha_prime(0) = -(1-eps) * pi/2 * sin(0) = 0."""
        t = torch.tensor(0.0)
        actual = cosine_schedule.alpha_prime(t)
        assert torch.isclose(actual, torch.tensor(0.0), atol=1e-7)

    def test_alpha_prime_at_t_half(self, cosine_schedule):
        t = torch.tensor(0.5)
        expected = -(1 - 1e-3) * (math.pi / 2) * math.sin(0.5 * math.pi / 2)
        assert torch.isclose(
            cosine_schedule.alpha_prime(t), torch.tensor(expected), atol=1e-5
        )

    def test_alpha_monotonically_decreasing(self, cosine_schedule):
        t = torch.linspace(0.0, 1.0, 200)
        alphas = cosine_schedule.alpha(t)
        diffs = alphas[1:] - alphas[:-1]
        assert (diffs <= 1e-6).all()

    def test_sigma_nonnegative(self, cosine_schedule):
        t = torch.linspace(0.0, 1.0, 200)
        sigmas = cosine_schedule.sigma(t)
        assert (sigmas >= -1e-6).all()

    def test_sigma_consistent_with_alpha(self, cosine_schedule):
        """exp(-sigma(t)) should equal alpha(t)."""
        t = torch.linspace(0.01, 0.99, 100)
        from_sigma = torch.exp(-cosine_schedule.sigma(t))
        from_alpha = cosine_schedule.alpha(t)
        assert torch.allclose(from_sigma, from_alpha, atol=1e-5)


class TestNoiseFactory:
    """Tests for get_noise factory function."""

    def test_get_noise_linear(self):
        cfg = OmegaConf.create({"type": "linear", "sigma_min": 0.0, "sigma_max": 10.0})
        schedule = get_noise(cfg)
        assert isinstance(schedule, LinearSchedule)

    def test_get_noise_loglinear(self):
        cfg = OmegaConf.create({"type": "loglinear", "eps": 1e-3})
        schedule = get_noise(cfg)
        assert isinstance(schedule, LogLinearSchedule)

    def test_get_noise_cosine(self):
        cfg = OmegaConf.create({"type": "cosine", "eps": 1e-3})
        schedule = get_noise(cfg)
        assert isinstance(schedule, CosineSchedule)

    def test_get_noise_invalid_type(self):
        cfg = OmegaConf.create({"type": "unknown"})
        with pytest.raises(ValueError, match="Unknown noise schedule type"):
            get_noise(cfg)


class TestScheduleTensorShapes:
    """Verify schedules handle various tensor shapes."""

    def test_batched_t(self, linear_schedule):
        t = torch.tensor([0.1, 0.5, 0.9])
        assert linear_schedule.alpha(t).shape == (3,)
        assert linear_schedule.sigma(t).shape == (3,)
        assert linear_schedule.alpha_prime(t).shape == (3,)

    def test_scalar_tensor(self, linear_schedule):
        t = torch.tensor(0.5)
        assert linear_schedule.alpha(t).shape == ()

    def test_2d_tensor(self, linear_schedule):
        t = torch.rand(4, 8)
        assert linear_schedule.alpha(t).shape == (4, 8)


# =========================================================================
# Forward Mask Tests
# =========================================================================


class TestForwardMask:
    """Tests for forward_mask function."""

    def test_output_shapes(self, sample_batch, linear_schedule, vocab_config):
        tokens = sample_batch["tokens"]
        pad_mask = sample_batch["pad_mask"]
        t = torch.rand(tokens.shape[0])

        x_t, mask_indicators = forward_mask(
            tokens, pad_mask, t, linear_schedule, vocab_config
        )
        assert x_t.shape == tokens.shape
        assert mask_indicators.shape == tokens.shape

    def test_output_dtypes(self, sample_batch, linear_schedule, vocab_config):
        tokens = sample_batch["tokens"]
        pad_mask = sample_batch["pad_mask"]
        t = torch.rand(tokens.shape[0])

        x_t, mask_indicators = forward_mask(
            tokens, pad_mask, t, linear_schedule, vocab_config
        )
        assert x_t.dtype == torch.long
        assert mask_indicators.dtype == torch.bool

    def test_pad_never_masked(self, sample_batch, linear_schedule, vocab_config):
        """PAD positions in x_t must match original PAD values."""
        tokens = sample_batch["tokens"]
        pad_mask = sample_batch["pad_mask"]
        t = torch.full((tokens.shape[0],), 0.5)

        x_t, mask_indicators = forward_mask(
            tokens, pad_mask, t, linear_schedule, vocab_config
        )

        # Where pad_mask is False (PAD), x_t should equal original tokens
        pad_positions = ~pad_mask
        assert (x_t[pad_positions] == tokens[pad_positions]).all()

    def test_mask_indicators_false_for_pad(
        self, sample_batch, linear_schedule, vocab_config
    ):
        """mask_indicators must be False wherever pad_mask is False."""
        tokens = sample_batch["tokens"]
        pad_mask = sample_batch["pad_mask"]
        t = torch.full((tokens.shape[0],), 0.5)

        _, mask_indicators = forward_mask(
            tokens, pad_mask, t, linear_schedule, vocab_config
        )

        assert not mask_indicators[~pad_mask].any()

    def test_correct_mask_token_types(
        self, sample_batch, linear_schedule, vocab_config
    ):
        """Node positions get NODE_MASK_IDX, edge positions get EDGE_MASK_IDX."""
        tokens = sample_batch["tokens"]
        pad_mask = sample_batch["pad_mask"]
        # Use high t to ensure many positions are masked
        t = torch.full((tokens.shape[0],), 0.99)

        x_t, mask_indicators = forward_mask(
            tokens, pad_mask, t, linear_schedule, vocab_config
        )

        n_max = vocab_config.n_max
        # Masked node positions should have NODE_MASK_IDX
        node_masked = mask_indicators[:, :n_max]
        assert (x_t[:, :n_max][node_masked] == NODE_MASK_IDX).all()

        # Masked edge positions should have EDGE_MASK_IDX
        edge_masked = mask_indicators[:, n_max:]
        assert (x_t[:, n_max:][edge_masked] == EDGE_MASK_IDX).all()

    def test_unmasked_positions_unchanged(
        self, sample_batch, linear_schedule, vocab_config
    ):
        """Non-masked real positions should keep their original values."""
        tokens = sample_batch["tokens"]
        pad_mask = sample_batch["pad_mask"]
        t = torch.full((tokens.shape[0],), 0.5)

        x_t, mask_indicators = forward_mask(
            tokens, pad_mask, t, linear_schedule, vocab_config
        )

        kept = pad_mask & ~mask_indicators
        assert (x_t[kept] == tokens[kept]).all()


# =========================================================================
# PAD Stress Tests
# =========================================================================


class TestPadStress:
    """PAD correctness stress tests — most critical tests in the project."""

    def test_pad_never_masked_stress_10k(self):
        """10,000 random masking operations across num_rooms 1-8.

        Verify EVERY PAD position remains unmasked EVERY time.
        """
        vc = RPLAN_VOCAB_CONFIG
        schedule = LinearSchedule(sigma_min=0.0, sigma_max=10.0)
        batch_size = 50

        for _ in range(200):  # 200 iterations * 50 batch = 10,000 samples
            # Random num_rooms per sample
            num_rooms_list = torch.randint(1, vc.n_max + 1, (batch_size,)).tolist()

            tokens = torch.zeros(batch_size, vc.seq_len, dtype=torch.long)
            pad_mask = torch.zeros(batch_size, vc.seq_len, dtype=torch.bool)

            for b, nr in enumerate(num_rooms_list):
                mask = vc.compute_pad_mask(nr)
                pad_mask[b] = mask
                # Fill real positions with random valid tokens
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

            t = torch.rand(batch_size)
            x_t, mask_indicators = forward_mask(tokens, pad_mask, t, schedule, vc)

            # CRITICAL: no PAD position was masked
            assert not mask_indicators[~pad_mask].any(), (
                "PAD position was masked!"
            )
            # PAD tokens unchanged
            assert (x_t[~pad_mask] == tokens[~pad_mask]).all(), (
                "PAD token was modified!"
            )

    def test_pad_never_masked_num_rooms_1(self):
        """Stress test with num_rooms=1: 35 out of 36 positions are PAD."""
        vc = RPLAN_VOCAB_CONFIG
        schedule = LinearSchedule()
        batch_size = 100

        pad_mask = vc.compute_pad_mask(1).unsqueeze(0).expand(batch_size, -1)
        tokens = torch.full((batch_size, vc.seq_len), NODE_PAD_IDX, dtype=torch.long)
        tokens[:, 0] = torch.randint(0, 13, (batch_size,))
        # Set edge PAD tokens
        for pos in range(vc.n_edges):
            tokens[:, vc.n_max + pos] = EDGE_PAD_IDX

        for _ in range(100):
            t = torch.rand(batch_size)
            x_t, mask_indicators = forward_mask(tokens, pad_mask, t, schedule, vc)
            assert not mask_indicators[~pad_mask].any()

    def test_pad_never_masked_num_rooms_2(self):
        """Stress test with num_rooms=2: 33 out of 36 positions are PAD."""
        vc = RPLAN_VOCAB_CONFIG
        schedule = LinearSchedule()
        batch_size = 100

        pad_mask = vc.compute_pad_mask(2).unsqueeze(0).expand(batch_size, -1)
        tokens = torch.full((batch_size, vc.seq_len), NODE_PAD_IDX, dtype=torch.long)
        tokens[:, 0] = torch.randint(0, 13, (batch_size,))
        tokens[:, 1] = torch.randint(0, 13, (batch_size,))
        # Edge PAD tokens
        for pos in range(vc.n_edges):
            seq_idx = vc.n_max + pos
            if pad_mask[0, seq_idx]:
                tokens[:, seq_idx] = torch.randint(0, 11, (batch_size,))
            else:
                tokens[:, seq_idx] = EDGE_PAD_IDX

        for _ in range(100):
            t = torch.rand(batch_size)
            x_t, mask_indicators = forward_mask(tokens, pad_mask, t, schedule, vc)
            assert not mask_indicators[~pad_mask].any()


# =========================================================================
# Masking Rate Tests
# =========================================================================


class TestMaskingRate:
    """Statistical tests that masking rate matches noise schedule."""

    def test_masking_rate_matches_alpha_statistically(self):
        """Over many samples, fraction masked ~ (1 - alpha(t))."""
        vc = RPLAN_VOCAB_CONFIG
        schedule = LinearSchedule()
        batch_size = 10000

        # All max-room graphs (no PAD) to simplify statistics
        pad_mask = vc.compute_pad_mask(vc.n_max).unsqueeze(0).expand(batch_size, -1)
        tokens = torch.randint(0, 11, (batch_size, vc.seq_len), dtype=torch.long)
        # Fix node tokens to valid range
        tokens[:, : vc.n_max] = torch.randint(0, 13, (batch_size, vc.n_max))

        t_val = 0.3
        t = torch.full((batch_size,), t_val)
        _, mask_indicators = forward_mask(tokens, pad_mask, t, schedule, vc)

        expected_rate = 1.0 - schedule.alpha(torch.tensor(t_val)).item()
        actual_rate = mask_indicators.float().mean().item()
        assert abs(actual_rate - expected_rate) < 0.02, (
            f"Expected masking rate ~{expected_rate:.3f}, got {actual_rate:.3f}"
        )

    def test_fully_masked_at_t1(self):
        """At t very close to 1, nearly all non-PAD positions are masked."""
        vc = RPLAN_VOCAB_CONFIG
        schedule = LinearSchedule()
        batch_size = 100

        pad_mask = vc.compute_pad_mask(vc.n_max).unsqueeze(0).expand(batch_size, -1)
        tokens = torch.randint(0, 11, (batch_size, vc.seq_len), dtype=torch.long)
        tokens[:, : vc.n_max] = torch.randint(0, 13, (batch_size, vc.n_max))

        t = torch.full((batch_size,), 1.0 - 1e-6)
        _, mask_indicators = forward_mask(tokens, pad_mask, t, schedule, vc)

        # alpha(~1) ~ exp(-10) ~ 4.5e-5, so nearly everything should be masked
        mask_rate = mask_indicators.float().mean().item()
        assert mask_rate > 0.99

    def test_nearly_clean_at_t0(self):
        """At t very close to 0, nearly all positions are unmasked."""
        vc = RPLAN_VOCAB_CONFIG
        schedule = LinearSchedule()
        batch_size = 100

        pad_mask = vc.compute_pad_mask(vc.n_max).unsqueeze(0).expand(batch_size, -1)
        tokens = torch.randint(0, 11, (batch_size, vc.seq_len), dtype=torch.long)
        tokens[:, : vc.n_max] = torch.randint(0, 13, (batch_size, vc.n_max))

        t = torch.full((batch_size,), 1e-6)
        _, mask_indicators = forward_mask(tokens, pad_mask, t, schedule, vc)

        mask_rate = mask_indicators.float().mean().item()
        assert mask_rate < 0.01
