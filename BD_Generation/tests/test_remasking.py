"""Tests for ReMDM-style remasking (bd_gen.diffusion.remasking)."""

from __future__ import annotations

import pytest
import torch

from bd_gen.data.vocab import (
    EDGE_MASK_IDX,
    EDGE_PAD_IDX,
    NODE_MASK_IDX,
    NODE_PAD_IDX,
)
from bd_gen.diffusion.remasking import (
    RemaskingSchedule,
    create_remasking_schedule,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cap_remasking(linear_schedule, vocab_config):
    """RemaskingSchedule with cap strategy, eta=0.1."""
    return RemaskingSchedule(
        strategy="cap",
        eta=0.1,
        noise_schedule=linear_schedule,
        vocab_config=vocab_config,
    )


@pytest.fixture
def rescale_remasking(linear_schedule, vocab_config):
    """RemaskingSchedule with rescale strategy, eta=0.5."""
    return RemaskingSchedule(
        strategy="rescale",
        eta=0.5,
        noise_schedule=linear_schedule,
        vocab_config=vocab_config,
    )


@pytest.fixture
def mixed_tokens_and_masks(vocab_config):
    """Create a batch with a mix of MASK, decoded, and PAD positions.

    Returns (x_t, pad_mask) where:
    - Sample 0: 3 rooms (some PAD)
    - Sample 1: 8 rooms (no PAD)
    """
    vc = vocab_config
    B = 2
    x_t = torch.zeros(B, vc.seq_len, dtype=torch.long)
    pad_mask = torch.zeros(B, vc.seq_len, dtype=torch.bool)

    # Sample 0: 3 rooms
    pm0 = vc.compute_pad_mask(3)
    pad_mask[0] = pm0
    # Nodes: 2 decoded, 1 MASK, rest PAD
    x_t[0, 0] = 2  # decoded (LivingRoom)
    x_t[0, 1] = 4  # decoded (Bathroom)
    x_t[0, 2] = NODE_MASK_IDX  # still masked
    for k in range(3, vc.n_max):
        x_t[0, k] = NODE_PAD_IDX
    # Edges: mix of decoded, MASK, PAD
    for pos in range(vc.n_edges):
        seq_idx = vc.n_max + pos
        if pm0[seq_idx]:
            if pos % 2 == 0:
                x_t[0, seq_idx] = 5  # decoded edge
            else:
                x_t[0, seq_idx] = EDGE_MASK_IDX
        else:
            x_t[0, seq_idx] = EDGE_PAD_IDX

    # Sample 1: 8 rooms (no PAD)
    pm1 = vc.compute_pad_mask(8)
    pad_mask[1] = pm1
    for k in range(vc.n_max):
        x_t[1, k] = k % 10  # all decoded
    for pos in range(vc.n_edges):
        seq_idx = vc.n_max + pos
        x_t[1, seq_idx] = pos % 10  # all decoded (no MASK, no PAD)

    return x_t, pad_mask


# ---------------------------------------------------------------------------
# Tests: Constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_invalid_strategy_raises(self, linear_schedule, vocab_config):
        with pytest.raises(ValueError, match="Unknown remasking strategy"):
            RemaskingSchedule("invalid", 0.1, linear_schedule, vocab_config)

    def test_negative_eta_raises(self, linear_schedule, vocab_config):
        with pytest.raises(ValueError, match="eta must be non-negative"):
            RemaskingSchedule("cap", -0.1, linear_schedule, vocab_config)

    def test_valid_cap(self, linear_schedule, vocab_config):
        rs = RemaskingSchedule("cap", 0.1, linear_schedule, vocab_config)
        assert rs.strategy == "cap"
        assert rs.eta == 0.1

    def test_valid_rescale(self, linear_schedule, vocab_config):
        rs = RemaskingSchedule("rescale", 0.5, linear_schedule, vocab_config)
        assert rs.strategy == "rescale"
        assert rs.eta == 0.5


# ---------------------------------------------------------------------------
# Tests: PAD protection (CRITICAL)
# ---------------------------------------------------------------------------


class TestPADProtection:
    def test_pad_never_remasked_stress(self, cap_remasking, vocab_config):
        """Stress test: 1000 calls, PAD positions NEVER become MASK."""
        vc = vocab_config
        torch.manual_seed(0)

        for _ in range(1000):
            B = 4
            num_rooms = torch.randint(1, vc.n_max + 1, (B,)).tolist()
            x_t = torch.zeros(B, vc.seq_len, dtype=torch.long)
            pad_mask = torch.zeros(B, vc.seq_len, dtype=torch.bool)

            for b in range(B):
                nr = num_rooms[b]
                pm = vc.compute_pad_mask(nr)
                pad_mask[b] = pm
                # Fill all real positions with decoded tokens
                for k in range(nr):
                    x_t[b, k] = k % 10
                for k in range(nr, vc.n_max):
                    x_t[b, k] = NODE_PAD_IDX
                for pos in range(vc.n_edges):
                    seq_idx = vc.n_max + pos
                    if pm[seq_idx]:
                        x_t[b, seq_idx] = pos % 10
                    else:
                        x_t[b, seq_idx] = EDGE_PAD_IDX

            result = cap_remasking(x_t, t_now=0.5, t_next=0.4, pad_mask=pad_mask)

            # Check: no PAD position changed
            for b in range(B):
                nr = num_rooms[b]
                for k in range(nr, vc.n_max):
                    assert result[b, k] == NODE_PAD_IDX, (
                        f"Node PAD at position {k} was remasked"
                    )
                for pos in range(vc.n_edges):
                    seq_idx = vc.n_max + pos
                    if not pad_mask[b, seq_idx]:
                        assert result[b, seq_idx] == EDGE_PAD_IDX, (
                            f"Edge PAD at position {seq_idx} was remasked"
                        )


# ---------------------------------------------------------------------------
# Tests: Correct MASK tokens per position type
# ---------------------------------------------------------------------------


class TestMaskTokens:
    def test_correct_mask_tokens(self, cap_remasking, mixed_tokens_and_masks,
                                  vocab_config):
        """Remasked nodes get NODE_MASK_IDX, remasked edges get EDGE_MASK_IDX."""
        x_t, pad_mask = mixed_tokens_and_masks
        vc = vocab_config

        # Use very high eta to guarantee some remasking
        high_eta_rs = RemaskingSchedule(
            "cap", 1.0, cap_remasking.noise_schedule, vc,
        )
        torch.manual_seed(42)
        result = high_eta_rs(x_t.clone(), t_now=0.5, t_next=0.4, pad_mask=pad_mask)

        n_max = vc.n_max
        for b in range(result.size(0)):
            for k in range(n_max):
                tok = result[b, k].item()
                # Must be either original value, NODE_MASK_IDX, or NODE_PAD_IDX
                assert tok in (x_t[b, k].item(), NODE_MASK_IDX, NODE_PAD_IDX)
                if tok != x_t[b, k].item() and pad_mask[b, k]:
                    assert tok == NODE_MASK_IDX
            for pos in range(vc.n_edges):
                seq_idx = n_max + pos
                tok = result[b, seq_idx].item()
                assert tok in (x_t[b, seq_idx].item(), EDGE_MASK_IDX, EDGE_PAD_IDX)
                if tok != x_t[b, seq_idx].item() and pad_mask[b, seq_idx]:
                    assert tok == EDGE_MASK_IDX


# ---------------------------------------------------------------------------
# Tests: Sigma formulas
# ---------------------------------------------------------------------------


class TestSigmaFormulas:
    def test_sigma_cap_formula(self, linear_schedule, vocab_config):
        """sigma_cap = min(eta, (1-alpha_s)/alpha_t)."""
        eta = 0.1
        rs = RemaskingSchedule("cap", eta, linear_schedule, vocab_config)

        t_now, t_next = 0.5, 0.4
        dev = torch.device("cpu")
        sigma = rs._compute_sigma_t(
            t_now, t_next, batch_size=1, device=dev,
        )

        # Compute expected
        alpha_t = linear_schedule.alpha(
            torch.tensor([t_now], dtype=torch.float64),
        )
        alpha_s = linear_schedule.alpha(
            torch.tensor([t_next], dtype=torch.float64),
        )
        sigma_max = ((1.0 - alpha_s) / (alpha_t + 1e-8)).clamp(0, 1)
        expected = min(eta, sigma_max.item())

        assert sigma.shape == (1, 1)
        assert abs(sigma.item() - expected) < 1e-5

    def test_sigma_rescale_formula(self, linear_schedule, vocab_config):
        """sigma_rescale = eta * min(1, (1-alpha_s)/alpha_t)."""
        eta = 0.5
        rs = RemaskingSchedule("rescale", eta, linear_schedule, vocab_config)

        t_now, t_next = 0.5, 0.4
        dev = torch.device("cpu")
        sigma = rs._compute_sigma_t(
            t_now, t_next, batch_size=1, device=dev,
        )

        alpha_t = linear_schedule.alpha(torch.tensor([t_now], dtype=torch.float64))
        alpha_s = linear_schedule.alpha(torch.tensor([t_next], dtype=torch.float64))
        sigma_max = ((1.0 - alpha_s) / (alpha_t + 1e-8)).clamp(0, 1)
        expected = eta * sigma_max.item()

        assert abs(sigma.item() - expected) < 1e-5

    def test_sigma_bounded_by_one(self, linear_schedule, vocab_config):
        """Sigma should never exceed 1.0."""
        rs = RemaskingSchedule("cap", 10.0, linear_schedule, vocab_config)
        # At high t_now, alpha_t is very small, sigma_max can be > 1
        sigma = rs._compute_sigma_t(0.99, 0.0, batch_size=1, device=torch.device("cpu"))
        assert sigma.item() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Tests: Remasking behavior
# ---------------------------------------------------------------------------


class TestRemaskingBehavior:
    def test_eta_zero_no_remasking(self, linear_schedule, vocab_config,
                                    mixed_tokens_and_masks):
        """eta=0 should never remask anything."""
        x_t, pad_mask = mixed_tokens_and_masks
        rs = RemaskingSchedule("cap", 0.0, linear_schedule, vocab_config)
        result = rs(x_t.clone(), t_now=0.5, t_next=0.4, pad_mask=pad_mask)
        assert torch.equal(result, x_t)

    def test_high_eta_remasks_most(self, linear_schedule, vocab_config):
        """eta=1.0 with moderate timesteps should remask most decoded positions."""
        vc = vocab_config
        rs = RemaskingSchedule("cap", 1.0, linear_schedule, vc)

        # All 8 rooms, all decoded
        B = 10
        x_t = torch.zeros(B, vc.seq_len, dtype=torch.long)
        pad_mask = torch.ones(B, vc.seq_len, dtype=torch.bool)
        for k in range(vc.n_max):
            x_t[:, k] = k % 10
        for pos in range(vc.n_edges):
            x_t[:, vc.n_max + pos] = pos % 10

        torch.manual_seed(0)
        result = rs(x_t.clone(), t_now=0.9, t_next=0.8, pad_mask=pad_mask)

        # Count remasked positions
        node_remasked = (result[:, :vc.n_max] == NODE_MASK_IDX).sum().item()
        edge_remasked = (result[:, vc.n_max:] == EDGE_MASK_IDX).sum().item()
        total_remasked = node_remasked + edge_remasked

        # With eta=1.0, sigma should be significant
        assert total_remasked > 0, "Expected some remasked positions"

    def test_only_unmasked_positions_remasked(self, cap_remasking,
                                               mixed_tokens_and_masks,
                                               vocab_config):
        """Already-MASK positions should never be double-masked."""
        x_t, pad_mask = mixed_tokens_and_masks
        vc = vocab_config

        is_node_mask_before = x_t[:, :vc.n_max] == NODE_MASK_IDX
        is_edge_mask_before = x_t[:, vc.n_max:] == EDGE_MASK_IDX

        torch.manual_seed(42)
        result = cap_remasking(x_t.clone(), t_now=0.5, t_next=0.4, pad_mask=pad_mask)

        # Positions that were MASK before should still be MASK
        is_node_mask_after = result[:, :vc.n_max] == NODE_MASK_IDX
        is_edge_mask_after = result[:, vc.n_max:] == EDGE_MASK_IDX

        assert (is_node_mask_before & ~is_node_mask_after).sum() == 0, (
            "A previously masked node position was unmasked by remasking"
        )
        assert (is_edge_mask_before & ~is_edge_mask_after).sum() == 0, (
            "A previously masked edge position was unmasked by remasking"
        )

    def test_does_not_modify_input(self, cap_remasking, mixed_tokens_and_masks):
        """Remasking should not modify the input tensor (uses clone internally)."""
        x_t, pad_mask = mixed_tokens_and_masks
        x_t_copy = x_t.clone()
        _ = cap_remasking(x_t, t_now=0.5, t_next=0.4, pad_mask=pad_mask)
        assert torch.equal(x_t, x_t_copy)


# ---------------------------------------------------------------------------
# Tests: Float64 precision
# ---------------------------------------------------------------------------


class TestFloat64Precision:
    def test_sigma_no_nan_at_extreme_timesteps(self, linear_schedule, vocab_config):
        """Sigma should be finite at extreme timesteps (near t=0 and t=1)."""
        rs = RemaskingSchedule("cap", 0.1, linear_schedule, vocab_config)

        # Near t=0
        sigma = rs._compute_sigma_t(0.01, 0.0, batch_size=1, device=torch.device("cpu"))
        assert torch.isfinite(sigma).all()

        # Near t=1
        dev = torch.device("cpu")
        sigma = rs._compute_sigma_t(0.99, 0.98, 1, device=dev)
        assert torch.isfinite(sigma).all()

        # Full range t=1 to t=0
        sigma = rs._compute_sigma_t(1.0, 0.0, batch_size=1, device=torch.device("cpu"))
        assert torch.isfinite(sigma).all()


# ---------------------------------------------------------------------------
# Tests: Factory function
# ---------------------------------------------------------------------------


class TestFactory:
    def test_disabled_returns_none(self, linear_schedule, vocab_config):
        config = {"enabled": False, "strategy": "cap", "eta": 0.1}
        result = create_remasking_schedule(config, linear_schedule, vocab_config)
        assert result is None

    def test_enabled_returns_schedule(self, linear_schedule, vocab_config):
        config = {"enabled": True, "strategy": "cap", "eta": 0.1}
        result = create_remasking_schedule(config, linear_schedule, vocab_config)
        assert isinstance(result, RemaskingSchedule)
        assert result.strategy == "cap"
        assert result.eta == 0.1

    def test_missing_enabled_key(self, linear_schedule, vocab_config):
        config = {"strategy": "cap", "eta": 0.1}
        result = create_remasking_schedule(config, linear_schedule, vocab_config)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: Integration with sample()
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_sample_with_remasking_produces_valid_output(
        self, dummy_model, linear_schedule, vocab_config,
    ):
        """Full sample() call with remasking produces tokens with no MASK."""
        from bd_gen.diffusion.sampling import sample

        remasking_fn = RemaskingSchedule(
            "cap", 0.1, linear_schedule, vocab_config,
        )

        tokens = sample(
            model=dummy_model,
            noise_schedule=linear_schedule,
            vocab_config=vocab_config,
            batch_size=4,
            num_steps=10,
            temperature=0.0,
            remasking_fn=remasking_fn,
            fixed_num_rooms=4,
            device="cpu",
        )

        assert tokens.shape == (4, vocab_config.seq_len)
        # No MASK tokens in output
        assert (tokens[:, :vocab_config.n_max] == NODE_MASK_IDX).sum() == 0
        assert (tokens[:, vocab_config.n_max:] == EDGE_MASK_IDX).sum() == 0

    def test_remasking_changes_output(self, dummy_model, linear_schedule,
                                       vocab_config):
        """Sampling with remasking should produce different results than without.

        Note: With a random (untrained) model, outputs may occasionally match.
        We use temperature > 0 and enough samples to make this extremely unlikely.
        """
        from bd_gen.diffusion.sampling import sample

        remasking_fn = RemaskingSchedule(
            "cap", 0.3, linear_schedule, vocab_config,
        )

        # Without remasking
        torch.manual_seed(42)
        tokens_baseline = sample(
            model=dummy_model, noise_schedule=linear_schedule,
            vocab_config=vocab_config, batch_size=8, num_steps=20,
            temperature=0.5, fixed_num_rooms=4, device="cpu",
        )

        # With remasking (same seed — but remasking adds extra randomness)
        torch.manual_seed(42)
        tokens_remasked = sample(
            model=dummy_model, noise_schedule=linear_schedule,
            vocab_config=vocab_config, batch_size=8, num_steps=20,
            temperature=0.5, remasking_fn=remasking_fn,
            fixed_num_rooms=4, device="cpu",
        )

        # They should differ (remasking consumes extra random numbers)
        assert not torch.equal(tokens_baseline, tokens_remasked), (
            "Remasking should change the output (extra stochasticity)"
        )
