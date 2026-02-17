"""Tests for MDLM ELBO loss.

Covers:
- Basic output properties (scalar, non-negative, finite, differentiable)
- PAD exclusion (zero contribution from PAD positions)
- Class weighting for edges and nodes
- Per-sample normalization by N_active
- ELBO weight w(t) computation and clamping
- Loss monotonicity (better predictions -> lower loss)
- Adversarial / collapse-detection cases from spec Section 7.4
"""

from __future__ import annotations

import torch

from bd_gen.data.vocab import (
    EDGE_NO_EDGE_IDX,
    EDGE_PAD_IDX,
    EDGE_VOCAB_SIZE,
    NODE_PAD_IDX,
    NODE_VOCAB_SIZE,
    RPLAN_VOCAB_CONFIG,
)
from bd_gen.diffusion.forward_process import forward_mask
from bd_gen.diffusion.loss import ELBOLoss
from bd_gen.diffusion.noise_schedule import LinearSchedule


def _make_loss_inputs(
    num_rooms_list: list[int],
    t_val: float = 0.5,
    vocab_config=RPLAN_VOCAB_CONFIG,
):
    """Helper: create a complete set of loss inputs for testing.

    Returns (node_logits, edge_logits, x0, x_t, pad_mask, mask_indicators, t).
    Logits are random; tokens have correct PAD placement.
    """
    vc = vocab_config
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

    schedule = LinearSchedule()
    t = torch.full((B,), t_val)
    x_t, mask_indicators = forward_mask(tokens, pad_mask, t, schedule, vc)

    node_logits = torch.randn(B, vc.n_max, NODE_VOCAB_SIZE, requires_grad=True)
    edge_logits = torch.randn(B, vc.n_edges, EDGE_VOCAB_SIZE, requires_grad=True)

    return node_logits, edge_logits, tokens, x_t, pad_mask, mask_indicators, t


# =========================================================================
# Basic Properties
# =========================================================================


class TestELBOLossBasic:
    """Core correctness tests for ELBOLoss."""

    def test_loss_scalar_output(self, elbo_loss, linear_schedule):
        nl, el, x0, xt, pm, mi, t = _make_loss_inputs([4, 6, 8])
        loss = elbo_loss(nl, el, x0, xt, pm, mi, t, linear_schedule)
        assert loss.dim() == 0

    def test_loss_nonnegative(self, elbo_loss, linear_schedule):
        nl, el, x0, xt, pm, mi, t = _make_loss_inputs([4, 6, 8])
        loss = elbo_loss(nl, el, x0, xt, pm, mi, t, linear_schedule)
        assert loss.item() >= 0

    def test_loss_finite(self, elbo_loss, linear_schedule):
        nl, el, x0, xt, pm, mi, t = _make_loss_inputs([4, 6, 8])
        loss = elbo_loss(nl, el, x0, xt, pm, mi, t, linear_schedule)
        assert torch.isfinite(loss)

    def test_loss_dtype(self, elbo_loss, linear_schedule):
        nl, el, x0, xt, pm, mi, t = _make_loss_inputs([4, 6, 8])
        loss = elbo_loss(nl, el, x0, xt, pm, mi, t, linear_schedule)
        assert loss.dtype == torch.float32

    def test_loss_requires_grad(self, elbo_loss, linear_schedule):
        nl, el, x0, xt, pm, mi, t = _make_loss_inputs([4, 6, 8])
        loss = elbo_loss(nl, el, x0, xt, pm, mi, t, linear_schedule)
        assert loss.requires_grad


# =========================================================================
# PAD Exclusion
# =========================================================================


class TestPadExclusion:
    """Verify PAD positions contribute zero loss."""

    def test_all_unmasked_loss_zero(self, elbo_loss, linear_schedule):
        """If no positions are masked, loss should be zero."""
        nl, el, x0, xt, pm, mi, t = _make_loss_inputs(
            [4, 6, 8, 8], t_val=1e-7
        )
        # Force mask_indicators to all False
        mi_zero = torch.zeros_like(mi)
        loss = elbo_loss(nl, el, x0, xt, pm, mi_zero, t, linear_schedule)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_single_room_loss_finite(self, elbo_loss, linear_schedule):
        """num_rooms=1: only 1 real position, loss must be finite."""
        nl, el, x0, xt, pm, mi, t = _make_loss_inputs([1])
        loss = elbo_loss(nl, el, x0, xt, pm, mi, t, linear_schedule)
        assert torch.isfinite(loss)

    def test_pad_tokens_unchanged_in_loss(self, linear_schedule):
        """Verify that changing PAD-position logits doesn't affect loss."""
        vc = RPLAN_VOCAB_CONFIG
        schedule = linear_schedule
        ecw = torch.ones(EDGE_VOCAB_SIZE)
        loss_fn = ELBOLoss(ecw, vocab_config=vc)

        nl, el, x0, xt, pm, mi, t = _make_loss_inputs([3])

        loss_1 = loss_fn(nl, el, x0, xt, pm, mi, t, schedule)

        # Modify logits at PAD positions
        nl2 = nl.clone().detach().requires_grad_(True)
        el2 = el.clone().detach().requires_grad_(True)
        nl2.data[:, 3:, :] = 999.0  # rooms 3-7 are PAD for num_rooms=3
        # Edge PAD positions
        for pos in range(vc.n_edges):
            i, j = vc.edge_position_to_pair(pos)
            if i >= 3 or j >= 3:
                el2.data[:, pos, :] = 999.0

        loss_2 = loss_fn(nl2, el2, x0, xt, pm, mi, t, schedule)
        assert torch.isclose(loss_1, loss_2, atol=1e-5)


# =========================================================================
# Class Weighting
# =========================================================================


class TestClassWeighting:
    """Verify class weighting applies correctly."""

    def test_edge_weights_affect_loss(self, linear_schedule):
        vc = RPLAN_VOCAB_CONFIG
        nl, el, x0, xt, pm, mi, t = _make_loss_inputs([8], t_val=0.5)

        ecw1 = torch.ones(EDGE_VOCAB_SIZE)
        ecw2 = torch.ones(EDGE_VOCAB_SIZE)
        ecw2[0] = 10.0  # Heavily weight left-above edges

        loss_fn1 = ELBOLoss(ecw1, vocab_config=vc)
        loss_fn2 = ELBOLoss(ecw2, vocab_config=vc)

        loss1 = loss_fn1(nl, el, x0, xt, pm, mi, t, linear_schedule)
        loss2 = loss_fn2(nl, el, x0, xt, pm, mi, t, linear_schedule)
        # Different weights should give different losses (unless no edges)
        # With num_rooms=8 and t=0.5, there will be masked edges
        assert not torch.isclose(loss1, loss2, atol=1e-6)

    def test_node_weights_optional(self, linear_schedule):
        """Loss works with node_class_weights=None."""
        vc = RPLAN_VOCAB_CONFIG
        ecw = torch.ones(EDGE_VOCAB_SIZE)
        loss_fn = ELBOLoss(ecw, node_class_weights=None, vocab_config=vc)
        nl, el, x0, xt, pm, mi, t = _make_loss_inputs([4, 8])
        loss = loss_fn(nl, el, x0, xt, pm, mi, t, linear_schedule)
        assert torch.isfinite(loss)

    def test_node_weights_affect_loss(self, linear_schedule):
        """When node_class_weights provided, changing them changes loss."""
        vc = RPLAN_VOCAB_CONFIG
        ecw = torch.ones(EDGE_VOCAB_SIZE)
        torch.manual_seed(123)
        nl, el, x0, xt, pm, mi, t = _make_loss_inputs([8], t_val=0.5)

        ncw1 = torch.ones(NODE_VOCAB_SIZE)
        ncw2 = torch.ones(NODE_VOCAB_SIZE)
        ncw2[0] = 10.0  # Heavily weight LivingRoom

        loss_fn1 = ELBOLoss(ecw, node_class_weights=ncw1, vocab_config=vc)
        loss_fn2 = ELBOLoss(ecw, node_class_weights=ncw2, vocab_config=vc)

        loss1 = loss_fn1(nl, el, x0, xt, pm, mi, t, linear_schedule)
        loss2 = loss_fn2(nl, el, x0, xt, pm, mi, t, linear_schedule)
        assert not torch.isclose(loss1, loss2, atol=1e-6)


# =========================================================================
# Per-Sample Normalization
# =========================================================================


class TestPerSampleNormalization:
    """Verify per-sample normalization by N_active."""

    def test_zero_masked_positions_no_nan(self, elbo_loss, linear_schedule):
        """When N_active=0 for a sample (e.g. t near 0), no NaN."""
        nl, el, x0, xt, pm, mi, t = _make_loss_inputs([4, 8], t_val=1e-7)
        mi_zero = torch.zeros_like(mi)
        loss = elbo_loss(nl, el, x0, xt, pm, mi_zero, t, linear_schedule)
        assert not torch.isnan(loss)


# =========================================================================
# ELBO Weight
# =========================================================================


class TestELBOWeight:
    """Verify w(t) computation and clamping."""

    def test_w_t_positive(self, elbo_loss, linear_schedule):
        """w(t) > 0 for all t in (0, 1]."""
        t = torch.linspace(0.01, 1.0, 100)
        w = elbo_loss._compute_w(t, linear_schedule)
        assert (w > 0).all()

    def test_w_t_clamped_near_zero(self, elbo_loss, linear_schedule):
        """At t=1e-6, w(t) is finite (clamped)."""
        t = torch.tensor([1e-6])
        w = elbo_loss._compute_w(t, linear_schedule)
        assert torch.isfinite(w).all()
        assert (w <= 1000.0).all()

    def test_w_t_no_nan(self, elbo_loss, linear_schedule):
        """w(t) is never NaN for any t in [0, 1]."""
        t = torch.linspace(0.0, 1.0, 1000)
        w = elbo_loss._compute_w(t, linear_schedule)
        assert not torch.isnan(w).any()

    def test_w_t_clamped_max(self, elbo_loss, linear_schedule):
        """w(t) is clamped to max 1000."""
        t = torch.tensor([0.0])
        w = elbo_loss._compute_w(t, linear_schedule)
        assert (w <= 1000.0).all()

    def test_w_t_float64_precision_near_zero(self, elbo_loss, linear_schedule):
        """w(t) near t_min benefits from float64; output is float32.

        At t=1e-5: alpha ≈ 0.9999, denominator 1-alpha ≈ 1e-4. Float64
        gives more accurate intermediate values. See arXiv:2409.02908.
        """
        t = torch.tensor([1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.5, 1.0])
        w = elbo_loss._compute_w(t, linear_schedule)
        assert torch.isfinite(w).all()
        assert (w > 0).all()
        # Output must be float32 (cast back from float64 internal)
        assert w.dtype == torch.float32


# =========================================================================
# Loss Decreases
# =========================================================================


class TestLossDecreases:
    """Verify loss is lower when logits match targets."""

    def test_matching_logits_lower_loss(self, linear_schedule):
        """Logits strongly predicting x0 targets -> lower loss than random."""
        vc = RPLAN_VOCAB_CONFIG
        ecw = torch.ones(EDGE_VOCAB_SIZE)
        loss_fn = ELBOLoss(ecw, vocab_config=vc)

        torch.manual_seed(42)
        nl_rand, el_rand, x0, xt, pm, mi, t = _make_loss_inputs(
            [8, 8, 8, 8], t_val=0.5
        )

        loss_random = loss_fn(nl_rand, el_rand, x0, xt, pm, mi, t, linear_schedule)

        # Create logits that strongly predict the correct targets
        nl_good = torch.zeros_like(nl_rand.data).requires_grad_(True)
        el_good = torch.zeros_like(el_rand.data).requires_grad_(True)
        for b in range(x0.shape[0]):
            for k in range(vc.n_max):
                nl_good.data[b, k, x0[b, k]] = 10.0
            for pos in range(vc.n_edges):
                el_good.data[b, pos, x0[b, vc.n_max + pos]] = 10.0

        loss_good = loss_fn(nl_good, el_good, x0, xt, pm, mi, t, linear_schedule)
        assert loss_good < loss_random


# =========================================================================
# Adversarial / Collapse-Detection Tests (Spec Section 7.4)
# =========================================================================


class TestAdversarialCases:
    """Adversarial and collapse-detection tests from spec Section 7.4."""

    def test_single_room_graph(self, linear_schedule):
        """num_rooms=1: loss computed on 1 position only."""
        ecw = torch.ones(EDGE_VOCAB_SIZE)
        loss_fn = ELBOLoss(ecw, vocab_config=RPLAN_VOCAB_CONFIG)
        nl, el, x0, xt, pm, mi, t = _make_loss_inputs([1], t_val=0.5)
        loss = loss_fn(nl, el, x0, xt, pm, mi, t, linear_schedule)
        assert torch.isfinite(loss)

    def test_two_room_graph(self, linear_schedule):
        """num_rooms=2: loss computed on 3 positions (2 nodes + 1 edge)."""
        ecw = torch.ones(EDGE_VOCAB_SIZE)
        loss_fn = ELBOLoss(ecw, vocab_config=RPLAN_VOCAB_CONFIG)
        nl, el, x0, xt, pm, mi, t = _make_loss_inputs([2], t_val=0.5)
        loss = loss_fn(nl, el, x0, xt, pm, mi, t, linear_schedule)
        assert torch.isfinite(loss)

    def test_max_room_graph(self, linear_schedule):
        """num_rooms=8: no PAD, all 36 positions real."""
        ecw = torch.ones(EDGE_VOCAB_SIZE)
        loss_fn = ELBOLoss(ecw, vocab_config=RPLAN_VOCAB_CONFIG)
        nl, el, x0, xt, pm, mi, t = _make_loss_inputs([8], t_val=0.5)
        loss = loss_fn(nl, el, x0, xt, pm, mi, t, linear_schedule)
        assert torch.isfinite(loss)

    def test_all_same_node_type(self, linear_schedule):
        """All nodes are LivingRoom (idx 0). Loss finite, no crash."""
        vc = RPLAN_VOCAB_CONFIG
        ecw = torch.ones(EDGE_VOCAB_SIZE)
        loss_fn = ELBOLoss(ecw, vocab_config=vc)

        tokens = torch.zeros(1, vc.seq_len, dtype=torch.long)
        pad_mask = vc.compute_pad_mask(vc.n_max).unsqueeze(0)
        tokens[0, : vc.n_max] = 0  # All LivingRoom
        tokens[0, vc.n_max :] = torch.randint(0, 11, (vc.n_edges,))

        schedule = linear_schedule
        t = torch.tensor([0.5])
        x_t, mi = forward_mask(tokens, pad_mask, t, schedule, vc)

        nl = torch.randn(1, vc.n_max, NODE_VOCAB_SIZE, requires_grad=True)
        el = torch.randn(1, vc.n_edges, EDGE_VOCAB_SIZE, requires_grad=True)

        loss = loss_fn(nl, el, tokens, x_t, pad_mask, mi, t, schedule)
        assert torch.isfinite(loss)

    def test_fully_connected_graph(self, linear_schedule):
        """All edges are spatial relationships (no no-edge). Loss finite."""
        vc = RPLAN_VOCAB_CONFIG
        ecw = torch.ones(EDGE_VOCAB_SIZE)
        loss_fn = ELBOLoss(ecw, vocab_config=vc)

        tokens = torch.zeros(1, vc.seq_len, dtype=torch.long)
        pad_mask = vc.compute_pad_mask(vc.n_max).unsqueeze(0)
        tokens[0, : vc.n_max] = torch.randint(0, 13, (vc.n_max,))
        tokens[0, vc.n_max :] = torch.randint(0, 10, (vc.n_edges,))  # Only spatial

        schedule = linear_schedule
        t = torch.tensor([0.5])
        x_t, mi = forward_mask(tokens, pad_mask, t, schedule, vc)

        nl = torch.randn(1, vc.n_max, NODE_VOCAB_SIZE, requires_grad=True)
        el = torch.randn(1, vc.n_edges, EDGE_VOCAB_SIZE, requires_grad=True)

        loss = loss_fn(nl, el, tokens, x_t, pad_mask, mi, t, schedule)
        assert torch.isfinite(loss)

    def test_all_no_edge_graph(self, linear_schedule):
        """All real edges are no-edge (idx 10). Loss finite."""
        vc = RPLAN_VOCAB_CONFIG
        ecw = torch.ones(EDGE_VOCAB_SIZE)
        loss_fn = ELBOLoss(ecw, vocab_config=vc)

        tokens = torch.zeros(1, vc.seq_len, dtype=torch.long)
        pad_mask = vc.compute_pad_mask(vc.n_max).unsqueeze(0)
        tokens[0, : vc.n_max] = torch.randint(0, 13, (vc.n_max,))
        tokens[0, vc.n_max :] = EDGE_NO_EDGE_IDX  # All no-edge

        schedule = linear_schedule
        t = torch.tensor([0.5])
        x_t, mi = forward_mask(tokens, pad_mask, t, schedule, vc)

        nl = torch.randn(1, vc.n_max, NODE_VOCAB_SIZE, requires_grad=True)
        el = torch.randn(1, vc.n_edges, EDGE_VOCAB_SIZE, requires_grad=True)

        loss = loss_fn(nl, el, tokens, x_t, pad_mask, mi, t, schedule)
        assert torch.isfinite(loss)

    def test_boundary_t_zero(self, linear_schedule):
        """t=1e-6: near-zero masked positions, loss=0 or near 0, no NaN."""
        ecw = torch.ones(EDGE_VOCAB_SIZE)
        loss_fn = ELBOLoss(ecw, vocab_config=RPLAN_VOCAB_CONFIG)
        nl, el, x0, xt, pm, mi, t = _make_loss_inputs([8], t_val=1e-6)
        loss = loss_fn(nl, el, x0, xt, pm, mi, t, linear_schedule)
        assert torch.isfinite(loss)

    def test_boundary_t_one(self, linear_schedule):
        """t=1-1e-6: everything masked, loss finite."""
        ecw = torch.ones(EDGE_VOCAB_SIZE)
        loss_fn = ELBOLoss(ecw, vocab_config=RPLAN_VOCAB_CONFIG)
        nl, el, x0, xt, pm, mi, t = _make_loss_inputs([8], t_val=1.0 - 1e-6)
        loss = loss_fn(nl, el, x0, xt, pm, mi, t, linear_schedule)
        assert torch.isfinite(loss)

    def test_extreme_batch_mix(self, linear_schedule):
        """Batch with num_rooms=[1, 2, 7, 8]. All processed correctly."""
        ecw = torch.ones(EDGE_VOCAB_SIZE)
        loss_fn = ELBOLoss(ecw, vocab_config=RPLAN_VOCAB_CONFIG)
        nl, el, x0, xt, pm, mi, t = _make_loss_inputs([1, 2, 7, 8], t_val=0.5)
        loss = loss_fn(nl, el, x0, xt, pm, mi, t, linear_schedule)
        assert torch.isfinite(loss)

    def test_gradient_magnitude_comparison(self, linear_schedule):
        """Compare ||grad_node|| vs ||grad_edge|| with class weights.

        Gradient magnitudes should be within 10x of each other.
        """
        vc = RPLAN_VOCAB_CONFIG
        ecw = torch.ones(EDGE_VOCAB_SIZE)
        ecw[EDGE_NO_EDGE_IDX] = 0.1  # Down-weight the dominant class
        loss_fn = ELBOLoss(ecw, vocab_config=vc)

        torch.manual_seed(0)
        nl, el, x0, xt, pm, mi, t = _make_loss_inputs(
            [8, 8, 8, 8], t_val=0.5
        )
        loss = loss_fn(nl, el, x0, xt, pm, mi, t, linear_schedule)
        loss.backward()

        grad_node_norm = nl.grad.norm().item()
        grad_edge_norm = el.grad.norm().item()

        # Both should be nonzero
        assert grad_node_norm > 0
        assert grad_edge_norm > 0

        # Ratio within 10x
        ratio = max(grad_node_norm, grad_edge_norm) / (
            min(grad_node_norm, grad_edge_norm) + 1e-10
        )
        assert ratio < 10, (
            f"Gradient imbalance: node={grad_node_norm:.4f}, "
            f"edge={grad_edge_norm:.4f}, ratio={ratio:.1f}"
        )
