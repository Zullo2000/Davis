"""Tests for soft violation computation (Phase G2).

Covers:
  - build_effective_probs correctness (spec tests 21-25)
  - Constraint soft violation convergence (spec tests 12, 14, 15, 17)
  - Constraint soft violation smoothness (spec test 13)
  - Constraint soft violation ranges (spec tests 16, 18)
  - Constraint soft violation edge cases (spec tests 19, 20)
  - hard_decode_x0 correctness
  - RewardComposer soft mode (extending spec tests 26-32)
  - triu_indices ordering sanity check
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from bd_gen.data.vocab import (
    EDGE_MASK_IDX,
    EDGE_NO_EDGE_IDX,
    EDGE_PAD_IDX,
    EDGE_VOCAB_SIZE,
    NODE_MASK_IDX,
    NODE_PAD_IDX,
    NODE_VOCAB_SIZE,
    RPLAN_VOCAB_CONFIG,
    VocabConfig,
)
from bd_gen.guidance.constraints import (
    CountRange,
    ExactCount,
    ForbidAdj,
    RequireAdj,
)
from bd_gen.guidance.reward import RewardComposer
from bd_gen.guidance.soft_violations import (
    build_effective_probs,
    build_effective_probs_batch,
    hard_decode_x0,
)


# =========================================================================
# Helpers
# =========================================================================


@pytest.fixture
def vc() -> VocabConfig:
    return RPLAN_VOCAB_CONFIG  # n_max=8, n_edges=28, seq_len=36


def _make_all_mask_tokens(vc: VocabConfig, num_rooms: int) -> torch.Tensor:
    """Build an x_t tensor where all real positions are MASK, rest are PAD."""
    x_t = torch.full((vc.seq_len,), NODE_PAD_IDX, dtype=torch.long)
    pad_mask = vc.compute_pad_mask(num_rooms)

    for k in range(num_rooms):
        x_t[k] = NODE_MASK_IDX
    for k in range(num_rooms, vc.n_max):
        x_t[k] = NODE_PAD_IDX

    for pos in range(vc.n_edges):
        seq_idx = vc.n_max + pos
        if pad_mask[seq_idx]:
            x_t[seq_idx] = EDGE_MASK_IDX
        else:
            x_t[seq_idx] = EDGE_PAD_IDX

    return x_t


def _make_committed_graph(vc: VocabConfig) -> tuple:
    """Build a fully committed graph (no MASK) for convergence tests.

    Graph: 4 rooms — Kitchen(2), LivingRoom(0), Bathroom(3), SecondRoom(7).
    Edge (0,1) = "above" (idx 3), Kitchen-LivingRoom adjacency.
    All other active edges = NO_EDGE.

    Returns: (x_t, node_logits, edge_logits, pad_mask, graph_dict)
    """
    num_rooms = 4
    pad_mask = vc.compute_pad_mask(num_rooms)

    x_t = torch.full((vc.seq_len,), NODE_PAD_IDX, dtype=torch.long)
    x_t[0] = 2  # Kitchen
    x_t[1] = 0  # LivingRoom
    x_t[2] = 3  # Bathroom
    x_t[3] = 7  # SecondRoom

    for pos in range(vc.n_edges):
        seq_idx = vc.n_max + pos
        if pad_mask[seq_idx]:
            i, j = vc.edge_position_to_pair(pos)
            if (i, j) == (0, 1):
                x_t[seq_idx] = 3  # "above" edge
            else:
                x_t[seq_idx] = EDGE_NO_EDGE_IDX
        else:
            x_t[seq_idx] = EDGE_PAD_IDX

    # Logits don't matter for committed positions (one-hot is used)
    node_logits = torch.randn(vc.n_max, NODE_VOCAB_SIZE, dtype=torch.float64)
    edge_logits = torch.randn(vc.n_edges, EDGE_VOCAB_SIZE, dtype=torch.float64)

    graph_dict = {
        "num_rooms": num_rooms,
        "node_types": [2, 0, 3, 7],
        "edge_triples": [(0, 1, 3)],
    }

    return x_t, node_logits, edge_logits, pad_mask, graph_dict


# =========================================================================
# Sanity: triu_indices ordering matches VocabConfig
# =========================================================================


class TestTriuIndicesOrdering:
    """Verify torch.triu_indices matches VocabConfig.edge_position_to_pair."""

    def test_ordering_matches(self, vc):
        i_indices, j_indices = torch.triu_indices(vc.n_max, vc.n_max, offset=1)
        assert len(i_indices) == vc.n_edges
        for pos in range(vc.n_edges):
            expected_i, expected_j = vc.edge_position_to_pair(pos)
            assert i_indices[pos].item() == expected_i
            assert j_indices[pos].item() == expected_j


# =========================================================================
# build_effective_probs tests (spec tests 21-25)
# =========================================================================


class TestBuildEffectiveProbs:
    """Tests 21-25: build_effective_probs correctness."""

    def test_committed_position_is_one_hot(self, vc):
        """(21) Non-MASK token -> one-hot probability vector."""
        x_t = torch.full((vc.seq_len,), NODE_PAD_IDX, dtype=torch.long)
        x_t[0] = 2  # Kitchen (committed)
        x_t[1] = NODE_MASK_IDX
        pad_mask = vc.compute_pad_mask(2)
        for pos in range(vc.n_edges):
            seq_idx = vc.n_max + pos
            x_t[seq_idx] = EDGE_NO_EDGE_IDX if pad_mask[seq_idx] else EDGE_PAD_IDX
        node_logits = torch.randn(vc.n_max, NODE_VOCAB_SIZE, dtype=torch.float64)
        edge_logits = torch.randn(vc.n_edges, EDGE_VOCAB_SIZE, dtype=torch.float64)

        node_probs, _ = build_effective_probs(x_t, node_logits, edge_logits, pad_mask, vc)

        expected = torch.zeros(NODE_VOCAB_SIZE, dtype=torch.float64)
        expected[2] = 1.0
        assert torch.allclose(node_probs[0], expected)

    def test_masked_position_is_softmax(self, vc):
        """(22) MASK token -> softmax(logits) probability vector."""
        x_t = torch.full((vc.seq_len,), NODE_PAD_IDX, dtype=torch.long)
        x_t[0] = NODE_MASK_IDX
        pad_mask = vc.compute_pad_mask(1)
        for pos in range(vc.n_edges):
            x_t[vc.n_max + pos] = EDGE_PAD_IDX
        node_logits = torch.randn(vc.n_max, NODE_VOCAB_SIZE, dtype=torch.float64)
        edge_logits = torch.randn(vc.n_edges, EDGE_VOCAB_SIZE, dtype=torch.float64)

        node_probs, _ = build_effective_probs(x_t, node_logits, edge_logits, pad_mask, vc)

        expected = F.softmax(node_logits[0].double(), dim=-1)
        assert torch.allclose(node_probs[0], expected)

    def test_pad_position_is_zeros(self, vc):
        """(23) PAD token -> all-zero probability vector."""
        x_t = torch.full((vc.seq_len,), NODE_PAD_IDX, dtype=torch.long)
        x_t[0] = 2  # 1 real room
        pad_mask = vc.compute_pad_mask(1)
        for pos in range(vc.n_edges):
            x_t[vc.n_max + pos] = EDGE_PAD_IDX
        node_logits = torch.randn(vc.n_max, NODE_VOCAB_SIZE, dtype=torch.float64)
        edge_logits = torch.randn(vc.n_edges, EDGE_VOCAB_SIZE, dtype=torch.float64)

        node_probs, _ = build_effective_probs(x_t, node_logits, edge_logits, pad_mask, vc)

        for k in range(1, vc.n_max):
            assert node_probs[k].abs().sum() == 0.0, f"Node {k} should be zeros"

    def test_active_positions_sum_to_one(self, vc):
        """(24) Active non-PAD positions have probs that sum to 1.0."""
        num_rooms = 4
        x_t = _make_all_mask_tokens(vc, num_rooms)
        pad_mask = vc.compute_pad_mask(num_rooms)
        node_logits = torch.randn(vc.n_max, NODE_VOCAB_SIZE, dtype=torch.float64)
        edge_logits = torch.randn(vc.n_edges, EDGE_VOCAB_SIZE, dtype=torch.float64)

        node_probs, edge_probs = build_effective_probs(
            x_t, node_logits, edge_logits, pad_mask, vc
        )

        # Active nodes sum to 1
        for k in range(num_rooms):
            assert node_probs[k].sum().item() == pytest.approx(1.0)
        # PAD nodes sum to 0
        for k in range(num_rooms, vc.n_max):
            assert node_probs[k].sum().item() == pytest.approx(0.0)

        # Active edges sum to 1
        for pos in range(vc.n_edges):
            if pad_mask[vc.n_max + pos]:
                assert edge_probs[pos].sum().item() == pytest.approx(1.0)
            else:
                assert edge_probs[pos].sum().item() == pytest.approx(0.0)

    def test_batch_matches_single(self, vc):
        """(25) Batched version matches per-sample version."""
        B = 3
        num_rooms_list = [2, 4, 6]

        x_t_list = [_make_all_mask_tokens(vc, nr) for nr in num_rooms_list]
        x_t_batch = torch.stack(x_t_list)
        pad_mask_batch = torch.stack([vc.compute_pad_mask(nr) for nr in num_rooms_list])

        torch.manual_seed(42)
        node_logits_batch = torch.randn(B, vc.n_max, NODE_VOCAB_SIZE, dtype=torch.float64)
        edge_logits_batch = torch.randn(B, vc.n_edges, EDGE_VOCAB_SIZE, dtype=torch.float64)

        node_probs_b, edge_probs_b = build_effective_probs_batch(
            x_t_batch, node_logits_batch, edge_logits_batch, pad_mask_batch, vc
        )

        for b in range(B):
            node_probs_s, edge_probs_s = build_effective_probs(
                x_t_batch[b], node_logits_batch[b], edge_logits_batch[b],
                pad_mask_batch[b], vc,
            )
            assert torch.allclose(node_probs_b[b], node_probs_s, atol=1e-12)
            assert torch.allclose(edge_probs_b[b], edge_probs_s, atol=1e-12)

    def test_dtype_is_float64(self, vc):
        """All outputs are float64 per spec Section 2.8."""
        x_t = _make_all_mask_tokens(vc, 4)
        pad_mask = vc.compute_pad_mask(4)
        node_logits = torch.randn(vc.n_max, NODE_VOCAB_SIZE)  # float32 input
        edge_logits = torch.randn(vc.n_edges, EDGE_VOCAB_SIZE)

        node_probs, edge_probs = build_effective_probs(
            x_t, node_logits, edge_logits, pad_mask, vc
        )

        assert node_probs.dtype == torch.float64
        assert edge_probs.dtype == torch.float64

    def test_edge_committed_is_one_hot(self, vc):
        """Edge committed positions produce one-hot vectors."""
        x_t = torch.full((vc.seq_len,), NODE_PAD_IDX, dtype=torch.long)
        x_t[0] = 2
        x_t[1] = 0
        pad_mask = vc.compute_pad_mask(2)
        # Edge (0,1) committed to "above" (idx 3)
        edge_01_pos = vc.pair_to_edge_position(0, 1)
        for pos in range(vc.n_edges):
            seq_idx = vc.n_max + pos
            if pad_mask[seq_idx]:
                x_t[seq_idx] = 3 if pos == edge_01_pos else EDGE_NO_EDGE_IDX
            else:
                x_t[seq_idx] = EDGE_PAD_IDX

        node_logits = torch.randn(vc.n_max, NODE_VOCAB_SIZE, dtype=torch.float64)
        edge_logits = torch.randn(vc.n_edges, EDGE_VOCAB_SIZE, dtype=torch.float64)

        _, edge_probs = build_effective_probs(x_t, node_logits, edge_logits, pad_mask, vc)

        expected = torch.zeros(EDGE_VOCAB_SIZE, dtype=torch.float64)
        expected[3] = 1.0
        assert torch.allclose(edge_probs[edge_01_pos], expected)


# =========================================================================
# hard_decode_x0 tests
# =========================================================================


class TestHardDecodeX0:
    """Tests for hard_decode_x0."""

    def test_committed_positions_unchanged(self, vc):
        """Committed positions keep their token after decode."""
        x_t, node_logits, edge_logits, pad_mask, _ = _make_committed_graph(vc)
        decoded = hard_decode_x0(x_t, node_logits, edge_logits, pad_mask, vc)
        # All non-PAD positions were committed, so decoded == x_t
        assert torch.equal(decoded, x_t)

    def test_mask_positions_use_argmax(self, vc):
        """MASK positions are replaced by argmax(logits)."""
        x_t = _make_all_mask_tokens(vc, 4)
        pad_mask = vc.compute_pad_mask(4)
        node_logits = torch.randn(vc.n_max, NODE_VOCAB_SIZE, dtype=torch.float64)
        edge_logits = torch.randn(vc.n_edges, EDGE_VOCAB_SIZE, dtype=torch.float64)

        decoded = hard_decode_x0(x_t, node_logits, edge_logits, pad_mask, vc)

        # Active node positions should have argmax of logits
        for k in range(4):
            assert decoded[k].item() == node_logits[k].argmax().item()

        # Active edge positions should have argmax of logits
        for pos in range(vc.n_edges):
            seq_idx = vc.n_max + pos
            if pad_mask[seq_idx]:
                assert decoded[seq_idx].item() == edge_logits[pos].argmax().item()

    def test_pad_positions_unchanged(self, vc):
        """PAD positions are left as-is."""
        x_t = _make_all_mask_tokens(vc, 4)
        pad_mask = vc.compute_pad_mask(4)
        node_logits = torch.randn(vc.n_max, NODE_VOCAB_SIZE, dtype=torch.float64)
        edge_logits = torch.randn(vc.n_edges, EDGE_VOCAB_SIZE, dtype=torch.float64)

        decoded = hard_decode_x0(x_t, node_logits, edge_logits, pad_mask, vc)

        for k in range(4, vc.n_max):
            assert decoded[k].item() == NODE_PAD_IDX
        for pos in range(vc.n_edges):
            seq_idx = vc.n_max + pos
            if not pad_mask[seq_idx]:
                assert decoded[seq_idx].item() == EDGE_PAD_IDX

    def test_no_mask_in_output(self, vc):
        """No MASK tokens remain in decoded output at real positions.

        Uses logits with MASK/PAD indices set to -inf to ensure argmax
        lands on a valid token (matching denoiser behavior).
        """
        x_t = _make_all_mask_tokens(vc, 4)
        pad_mask = vc.compute_pad_mask(4)

        # Build logits where MASK and PAD indices are -inf (matching denoiser)
        node_logits = torch.randn(vc.n_max, NODE_VOCAB_SIZE, dtype=torch.float64)
        node_logits[:, NODE_MASK_IDX] = float("-inf")
        node_logits[:, NODE_PAD_IDX] = float("-inf")
        edge_logits = torch.randn(vc.n_edges, EDGE_VOCAB_SIZE, dtype=torch.float64)
        edge_logits[:, EDGE_MASK_IDX] = float("-inf")
        edge_logits[:, EDGE_PAD_IDX] = float("-inf")

        decoded = hard_decode_x0(x_t, node_logits, edge_logits, pad_mask, vc)

        for k in range(4):
            assert decoded[k].item() != NODE_MASK_IDX
        for pos in range(vc.n_edges):
            seq_idx = vc.n_max + pos
            if pad_mask[seq_idx]:
                assert decoded[seq_idx].item() != EDGE_MASK_IDX

    def test_batched(self, vc):
        """Batched version works correctly."""
        B = 2
        x_t = torch.stack([_make_all_mask_tokens(vc, 3), _make_all_mask_tokens(vc, 5)])
        pad_mask = torch.stack([vc.compute_pad_mask(3), vc.compute_pad_mask(5)])
        node_logits = torch.randn(B, vc.n_max, NODE_VOCAB_SIZE, dtype=torch.float64)
        edge_logits = torch.randn(B, vc.n_edges, EDGE_VOCAB_SIZE, dtype=torch.float64)

        decoded = hard_decode_x0(x_t, node_logits, edge_logits, pad_mask, vc)

        assert decoded.shape == (B, vc.seq_len)
        # Verify first sample
        for k in range(3):
            assert decoded[0, k].item() == node_logits[0, k].argmax().item()
        # Verify PAD unchanged
        for k in range(3, vc.n_max):
            assert decoded[0, k].item() == NODE_PAD_IDX

    def test_does_not_mutate_input(self, vc):
        """hard_decode_x0 does not modify the input tensor."""
        x_t = _make_all_mask_tokens(vc, 4)
        x_t_orig = x_t.clone()
        pad_mask = vc.compute_pad_mask(4)
        node_logits = torch.randn(vc.n_max, NODE_VOCAB_SIZE, dtype=torch.float64)
        edge_logits = torch.randn(vc.n_edges, EDGE_VOCAB_SIZE, dtype=torch.float64)

        _ = hard_decode_x0(x_t, node_logits, edge_logits, pad_mask, vc)
        assert torch.equal(x_t, x_t_orig)


# =========================================================================
# Soft violation convergence tests (spec tests 12, 14, 15, 17)
# =========================================================================


class TestSoftConvergence:
    """When all positions are committed (one-hot), soft violation == hard violation."""

    def test_exact_count_soft_converges(self, vc):
        """(12) ExactCount: one-hot probs -> soft == hard."""
        x_t, node_logits, edge_logits, pad_mask, graph_dict = _make_committed_graph(vc)
        node_probs, edge_probs = build_effective_probs(
            x_t, node_logits, edge_logits, pad_mask, vc
        )

        # Kitchen(2) count=1, target=1 -> v=0
        c = ExactCount(name="1_kitchen", room_type_idx=2, target=1)
        hard = c.hard_violation(graph_dict).violation
        soft = c.soft_violation(node_probs, edge_probs, pad_mask, vc)
        assert soft.item() == pytest.approx(hard)

        # Kitchen(2) count=1, target=3 -> v=2
        c2 = ExactCount(name="3_kitchen", room_type_idx=2, target=3)
        hard2 = c2.hard_violation(graph_dict).violation
        soft2 = c2.soft_violation(node_probs, edge_probs, pad_mask, vc)
        assert soft2.item() == pytest.approx(hard2)

    def test_count_range_soft_converges(self, vc):
        """(14) CountRange: one-hot probs -> soft == hard."""
        x_t, node_logits, edge_logits, pad_mask, graph_dict = _make_committed_graph(vc)
        node_probs, edge_probs = build_effective_probs(
            x_t, node_logits, edge_logits, pad_mask, vc
        )

        # SecondRoom(7) count=1, range [1,4] -> v=0 (satisfied)
        c = CountRange(name="sr_range", room_type_idx=7, lo=1, hi=4)
        hard = c.hard_violation(graph_dict).violation
        soft = c.soft_violation(node_probs, edge_probs, pad_mask, vc)
        assert soft.item() == pytest.approx(hard)

        # SecondRoom(7) count=1, range [2,4] -> v=1 (under)
        c2 = CountRange(name="sr_range2", room_type_idx=7, lo=2, hi=4)
        hard2 = c2.hard_violation(graph_dict).violation
        soft2 = c2.soft_violation(node_probs, edge_probs, pad_mask, vc)
        assert soft2.item() == pytest.approx(hard2)

    def test_require_adj_soft_converges(self, vc):
        """(15) RequireAdj: one-hot probs -> soft == hard."""
        x_t, node_logits, edge_logits, pad_mask, graph_dict = _make_committed_graph(vc)
        node_probs, edge_probs = build_effective_probs(
            x_t, node_logits, edge_logits, pad_mask, vc
        )

        # Kitchen(2)-LivingRoom(0) edge exists -> v=0
        c = RequireAdj(name="kitchen_adj_living", type_a_idx=2, type_b_idx=0)
        hard = c.hard_violation(graph_dict).violation
        soft = c.soft_violation(node_probs, edge_probs, pad_mask, vc)
        assert soft.item() == pytest.approx(hard, abs=1e-10)

        # Kitchen(2)-SecondRoom(7): no edge -> v=1
        c2 = RequireAdj(name="kitchen_adj_sr", type_a_idx=2, type_b_idx=7)
        hard2 = c2.hard_violation(graph_dict).violation
        soft2 = c2.soft_violation(node_probs, edge_probs, pad_mask, vc)
        assert soft2.item() == pytest.approx(hard2, abs=1e-10)

    def test_forbid_adj_soft_converges(self, vc):
        """(17) ForbidAdj: one-hot probs -> soft == hard."""
        x_t, node_logits, edge_logits, pad_mask, graph_dict = _make_committed_graph(vc)
        node_probs, edge_probs = build_effective_probs(
            x_t, node_logits, edge_logits, pad_mask, vc
        )

        # Forbid Bathroom(3)-Kitchen(2): no such edge -> v=0
        c = ForbidAdj(name="no_bath_kitchen", type_a_idx=3, type_b_idx=2)
        hard = c.hard_violation(graph_dict).violation
        soft = c.soft_violation(node_probs, edge_probs, pad_mask, vc)
        assert soft.item() == pytest.approx(hard, abs=1e-10)

        # Forbid Kitchen(2)-LivingRoom(0): edge exists -> v=1
        c2 = ForbidAdj(name="forbid_kl", type_a_idx=2, type_b_idx=0)
        hard2 = c2.hard_violation(graph_dict).violation
        soft2 = c2.soft_violation(node_probs, edge_probs, pad_mask, vc)
        assert soft2.item() == pytest.approx(hard2, abs=1e-10)


# =========================================================================
# Soft violation smoothness (spec test 13)
# =========================================================================


class TestSoftSmoothness:
    """(13) Small logit perturbation -> small violation change."""

    def test_exact_count_soft_smooth(self, vc):
        num_rooms = 4
        x_t = _make_all_mask_tokens(vc, num_rooms)
        pad_mask = vc.compute_pad_mask(num_rooms)
        torch.manual_seed(42)
        node_logits = torch.randn(vc.n_max, NODE_VOCAB_SIZE, dtype=torch.float64)
        edge_logits = torch.randn(vc.n_edges, EDGE_VOCAB_SIZE, dtype=torch.float64)

        c = ExactCount(name="1_kitchen", room_type_idx=2, target=1)

        node_probs, edge_probs = build_effective_probs(
            x_t, node_logits, edge_logits, pad_mask, vc
        )
        v_base = c.soft_violation(node_probs, edge_probs, pad_mask, vc)

        eps = 0.01
        node_logits_p = node_logits + eps * torch.randn_like(node_logits)
        node_probs_p, edge_probs_p = build_effective_probs(
            x_t, node_logits_p, edge_logits, pad_mask, vc
        )
        v_perturbed = c.soft_violation(node_probs_p, edge_probs_p, pad_mask, vc)

        delta = abs(v_perturbed.item() - v_base.item())
        assert delta < 1.0, f"Violation jumped by {delta} for eps={eps}"

    def test_require_adj_soft_smooth(self, vc):
        num_rooms = 4
        x_t = _make_all_mask_tokens(vc, num_rooms)
        pad_mask = vc.compute_pad_mask(num_rooms)
        torch.manual_seed(123)
        node_logits = torch.randn(vc.n_max, NODE_VOCAB_SIZE, dtype=torch.float64)
        edge_logits = torch.randn(vc.n_edges, EDGE_VOCAB_SIZE, dtype=torch.float64)

        c = RequireAdj(name="req", type_a_idx=2, type_b_idx=0)

        node_probs, edge_probs = build_effective_probs(
            x_t, node_logits, edge_logits, pad_mask, vc
        )
        v_base = c.soft_violation(node_probs, edge_probs, pad_mask, vc)

        eps = 0.01
        node_logits_p = node_logits + eps * torch.randn_like(node_logits)
        edge_logits_p = edge_logits + eps * torch.randn_like(edge_logits)
        node_probs_p, edge_probs_p = build_effective_probs(
            x_t, node_logits_p, edge_logits_p, pad_mask, vc
        )
        v_perturbed = c.soft_violation(node_probs_p, edge_probs_p, pad_mask, vc)

        delta = abs(v_perturbed.item() - v_base.item())
        assert delta < 0.5, f"Violation jumped by {delta} for eps={eps}"


# =========================================================================
# Soft violation range tests (spec tests 16, 18)
# =========================================================================


class TestSoftViolationRanges:
    """Test that soft violations are in expected ranges."""

    def test_require_adj_soft_range(self, vc):
        """(16) RequireAdj soft violation is in [0, 1]."""
        for seed in [0, 42, 123]:
            torch.manual_seed(seed)
            x_t = _make_all_mask_tokens(vc, 4)
            pad_mask = vc.compute_pad_mask(4)
            node_logits = torch.randn(vc.n_max, NODE_VOCAB_SIZE, dtype=torch.float64)
            edge_logits = torch.randn(vc.n_edges, EDGE_VOCAB_SIZE, dtype=torch.float64)
            node_probs, edge_probs = build_effective_probs(
                x_t, node_logits, edge_logits, pad_mask, vc
            )

            c = RequireAdj(name="req", type_a_idx=2, type_b_idx=0)
            v = c.soft_violation(node_probs, edge_probs, pad_mask, vc)
            assert 0.0 <= v.item() <= 1.0 + 1e-10, f"seed={seed}, v={v.item()}"

    def test_forbid_adj_soft_non_negative(self, vc):
        """(18) ForbidAdj soft violation is always >= 0."""
        for seed in [0, 42, 123]:
            torch.manual_seed(seed)
            x_t = _make_all_mask_tokens(vc, 4)
            pad_mask = vc.compute_pad_mask(4)
            node_logits = torch.randn(vc.n_max, NODE_VOCAB_SIZE, dtype=torch.float64)
            edge_logits = torch.randn(vc.n_edges, EDGE_VOCAB_SIZE, dtype=torch.float64)
            node_probs, edge_probs = build_effective_probs(
                x_t, node_logits, edge_logits, pad_mask, vc
            )

            c = ForbidAdj(name="forbid", type_a_idx=3, type_b_idx=2)
            v = c.soft_violation(node_probs, edge_probs, pad_mask, vc)
            assert v.item() >= -1e-15, f"seed={seed}, v={v.item()}"

    def test_exact_count_soft_non_negative(self, vc):
        """ExactCount soft violation is always >= 0."""
        torch.manual_seed(0)
        x_t = _make_all_mask_tokens(vc, 4)
        pad_mask = vc.compute_pad_mask(4)
        node_logits = torch.randn(vc.n_max, NODE_VOCAB_SIZE, dtype=torch.float64)
        edge_logits = torch.randn(vc.n_edges, EDGE_VOCAB_SIZE, dtype=torch.float64)
        node_probs, edge_probs = build_effective_probs(
            x_t, node_logits, edge_logits, pad_mask, vc
        )

        c = ExactCount(name="ec", room_type_idx=2, target=1)
        v = c.soft_violation(node_probs, edge_probs, pad_mask, vc)
        assert v.item() >= -1e-15

    def test_count_range_soft_non_negative(self, vc):
        """CountRange soft violation is always >= 0."""
        torch.manual_seed(0)
        x_t = _make_all_mask_tokens(vc, 4)
        pad_mask = vc.compute_pad_mask(4)
        node_logits = torch.randn(vc.n_max, NODE_VOCAB_SIZE, dtype=torch.float64)
        edge_logits = torch.randn(vc.n_edges, EDGE_VOCAB_SIZE, dtype=torch.float64)
        node_probs, edge_probs = build_effective_probs(
            x_t, node_logits, edge_logits, pad_mask, vc
        )

        c = CountRange(name="cr", room_type_idx=7, lo=1, hi=4)
        v = c.soft_violation(node_probs, edge_probs, pad_mask, vc)
        assert v.item() >= -1e-15


# =========================================================================
# Soft violation edge cases (spec tests 19, 20)
# =========================================================================


class TestSoftViolationEdgeCases:
    """Edge cases: all-masked and PAD exclusion."""

    def test_all_masked_sensible(self, vc):
        """(19) All positions MASK -> violations are finite and >= 0."""
        x_t = _make_all_mask_tokens(vc, 8)  # all 8 rooms, all MASK
        pad_mask = vc.compute_pad_mask(8)
        torch.manual_seed(0)
        node_logits = torch.randn(vc.n_max, NODE_VOCAB_SIZE, dtype=torch.float64)
        edge_logits = torch.randn(vc.n_edges, EDGE_VOCAB_SIZE, dtype=torch.float64)
        node_probs, edge_probs = build_effective_probs(
            x_t, node_logits, edge_logits, pad_mask, vc
        )

        constraints = [
            ExactCount(name="ec", room_type_idx=2, target=1),
            CountRange(name="cr", room_type_idx=7, lo=1, hi=4),
            RequireAdj(name="ra", type_a_idx=2, type_b_idx=0),
            ForbidAdj(name="fa", type_a_idx=3, type_b_idx=2),
        ]
        for c in constraints:
            v = c.soft_violation(node_probs, edge_probs, pad_mask, vc)
            assert torch.isfinite(v), f"{c.name}: not finite: {v}"
            assert v.item() >= 0.0, f"{c.name}: negative: {v}"

    def test_pad_positions_excluded(self, vc):
        """(20) PAD positions are already zero; explicit re-zeroing doesn't change result."""
        num_rooms = 4
        x_t = _make_all_mask_tokens(vc, num_rooms)
        pad_mask = vc.compute_pad_mask(num_rooms)
        torch.manual_seed(0)
        node_logits = torch.randn(vc.n_max, NODE_VOCAB_SIZE, dtype=torch.float64)
        edge_logits = torch.randn(vc.n_edges, EDGE_VOCAB_SIZE, dtype=torch.float64)
        node_probs, edge_probs = build_effective_probs(
            x_t, node_logits, edge_logits, pad_mask, vc
        )

        c = ExactCount(name="ec", room_type_idx=2, target=1)
        v1 = c.soft_violation(node_probs, edge_probs, pad_mask, vc)

        # Verify PAD node positions are indeed zero
        assert node_probs[num_rooms:].abs().sum() == 0.0

        # Explicitly zero PADs again — result should be identical
        node_probs_copy = node_probs.clone()
        node_probs_copy[~pad_mask[: vc.n_max]] = 0.0
        v2 = c.soft_violation(node_probs_copy, edge_probs, pad_mask, vc)
        assert v1.item() == pytest.approx(v2.item())

    def test_single_room_require_adj(self, vc):
        """With 1 room, no edges exist. RequireAdj violation = 1.0."""
        x_t = _make_all_mask_tokens(vc, 1)
        pad_mask = vc.compute_pad_mask(1)
        node_logits = torch.randn(vc.n_max, NODE_VOCAB_SIZE, dtype=torch.float64)
        edge_logits = torch.randn(vc.n_edges, EDGE_VOCAB_SIZE, dtype=torch.float64)
        node_probs, edge_probs = build_effective_probs(
            x_t, node_logits, edge_logits, pad_mask, vc
        )

        c = RequireAdj(name="req", type_a_idx=2, type_b_idx=0)
        v = c.soft_violation(node_probs, edge_probs, pad_mask, vc)
        # All edge positions are PAD -> p_ij=0 -> P(exists)=0 -> v=1
        assert v.item() == pytest.approx(1.0)


# =========================================================================
# RewardComposer soft mode tests (extending spec tests 26-32)
# =========================================================================


class TestRewardComposerSoftMode:
    """RewardComposer: soft mode energy/reward tests."""

    def test_soft_energy_non_negative(self, vc):
        """(26 soft) Soft energy should be >= 0."""
        x_t = _make_all_mask_tokens(vc, 4)
        pad_mask = vc.compute_pad_mask(4)
        torch.manual_seed(0)
        node_logits = torch.randn(vc.n_max, NODE_VOCAB_SIZE, dtype=torch.float64)
        edge_logits = torch.randn(vc.n_edges, EDGE_VOCAB_SIZE, dtype=torch.float64)
        node_probs, edge_probs = build_effective_probs(
            x_t, node_logits, edge_logits, pad_mask, vc
        )

        constraints = [ExactCount(name="ec", room_type_idx=2, target=1)]
        composer = RewardComposer(constraints=constraints, phi="linear")
        energy, details = composer.compute_energy_soft(
            node_probs, edge_probs, pad_mask, vc
        )
        assert energy.item() >= 0.0

    def test_soft_reward_non_positive(self, vc):
        """(27 soft) Soft reward should be <= 0."""
        x_t = _make_all_mask_tokens(vc, 4)
        pad_mask = vc.compute_pad_mask(4)
        torch.manual_seed(0)
        node_logits = torch.randn(vc.n_max, NODE_VOCAB_SIZE, dtype=torch.float64)
        edge_logits = torch.randn(vc.n_edges, EDGE_VOCAB_SIZE, dtype=torch.float64)
        node_probs, edge_probs = build_effective_probs(
            x_t, node_logits, edge_logits, pad_mask, vc
        )

        constraints = [ExactCount(name="ec", room_type_idx=2, target=1)]
        composer = RewardComposer(constraints=constraints, phi="linear")
        reward, _ = composer.compute_reward_soft(
            node_probs, edge_probs, pad_mask, vc
        )
        assert reward.item() <= 0.0

    def test_soft_perfect_graph_zero_energy(self, vc):
        """(28 soft) All constraints satisfied on committed graph -> E=0."""
        x_t, node_logits, edge_logits, pad_mask, _ = _make_committed_graph(vc)
        node_probs, edge_probs = build_effective_probs(
            x_t, node_logits, edge_logits, pad_mask, vc
        )

        constraints = [
            ExactCount(name="1_kitchen", room_type_idx=2, target=1),
            RequireAdj(name="kitchen_adj_living", type_a_idx=2, type_b_idx=0),
            ForbidAdj(name="no_bath_kitchen", type_a_idx=3, type_b_idx=2),
        ]
        composer = RewardComposer(constraints=constraints, phi="linear")
        energy, details = composer.compute_energy_soft(
            node_probs, edge_probs, pad_mask, vc
        )
        assert energy.item() == pytest.approx(0.0, abs=1e-10)

    def test_soft_phi_quadratic(self, vc):
        """(29 soft) phi='quadratic' applies correctly to soft violations."""
        x_t, node_logits, edge_logits, pad_mask, graph_dict = _make_committed_graph(vc)
        node_probs, edge_probs = build_effective_probs(
            x_t, node_logits, edge_logits, pad_mask, vc
        )

        # Kitchen(2) count=1, target=3 -> v=2, phi(v) = v^2 = 4
        c = ExactCount(name="kitchen", room_type_idx=2, target=3, weight=1.0)
        composer = RewardComposer(constraints=[c], phi="quadratic")
        energy, _ = composer.compute_energy_soft(
            node_probs, edge_probs, pad_mask, vc
        )
        assert energy.item() == pytest.approx(4.0, abs=1e-10)

    def test_soft_phi_log1p(self, vc):
        """(30 soft) phi='log1p' applies correctly to soft violations."""
        x_t, node_logits, edge_logits, pad_mask, graph_dict = _make_committed_graph(vc)
        node_probs, edge_probs = build_effective_probs(
            x_t, node_logits, edge_logits, pad_mask, vc
        )

        # Kitchen(2) count=1, target=3 -> v=2, phi(v) = log(1+2) = log(3)
        c = ExactCount(name="kitchen", room_type_idx=2, target=3, weight=1.0)
        composer = RewardComposer(constraints=[c], phi="log1p")
        energy, _ = composer.compute_energy_soft(
            node_probs, edge_probs, pad_mask, vc
        )
        assert energy.item() == pytest.approx(math.log(3), abs=1e-10)

    def test_soft_matches_hard_on_committed(self, vc):
        """On fully committed graph, soft energy == hard energy."""
        x_t, node_logits, edge_logits, pad_mask, graph_dict = _make_committed_graph(vc)
        node_probs, edge_probs = build_effective_probs(
            x_t, node_logits, edge_logits, pad_mask, vc
        )

        constraints = [
            ExactCount(name="1_kitchen", room_type_idx=2, target=1, weight=1.0),
            CountRange(name="sr_range", room_type_idx=7, lo=2, hi=4, weight=1.0),
            RequireAdj(name="kitchen_adj_living", type_a_idx=2, type_b_idx=0, weight=1.0),
            ForbidAdj(name="no_bath_kitchen", type_a_idx=3, type_b_idx=2, weight=1.0),
        ]
        composer = RewardComposer(constraints=constraints, phi="linear")

        hard_energy, _ = composer.compute_energy_hard(graph_dict)
        soft_energy, _ = composer.compute_energy_soft(
            node_probs, edge_probs, pad_mask, vc
        )
        assert soft_energy.item() == pytest.approx(hard_energy, abs=1e-10)

    def test_soft_calibration_effect(self, vc):
        """(32 soft) Calibration normalizer affects soft energy correctly."""
        x_t, node_logits, edge_logits, pad_mask, _ = _make_committed_graph(vc)
        node_probs, edge_probs = build_effective_probs(
            x_t, node_logits, edge_logits, pad_mask, vc
        )

        # Kitchen count=1, target=3 -> v=2, p90=2 -> normalized = (1/2)*2 = 1
        c = ExactCount(name="kitchen", room_type_idx=2, target=3, weight=1.0)
        c.p90_normalizer = 2.0
        composer = RewardComposer(constraints=[c], phi="linear")
        energy, _ = composer.compute_energy_soft(
            node_probs, edge_probs, pad_mask, vc
        )
        assert energy.item() == pytest.approx(1.0, abs=1e-10)

    def test_soft_energy_dtype_float64(self, vc):
        """Soft energy output should be float64."""
        x_t = _make_all_mask_tokens(vc, 4)
        pad_mask = vc.compute_pad_mask(4)
        node_logits = torch.randn(vc.n_max, NODE_VOCAB_SIZE, dtype=torch.float64)
        edge_logits = torch.randn(vc.n_edges, EDGE_VOCAB_SIZE, dtype=torch.float64)
        node_probs, edge_probs = build_effective_probs(
            x_t, node_logits, edge_logits, pad_mask, vc
        )

        constraints = [ExactCount(name="ec", room_type_idx=2, target=1)]
        composer = RewardComposer(constraints=constraints, phi="linear")
        energy, details = composer.compute_energy_soft(
            node_probs, edge_probs, pad_mask, vc
        )
        assert energy.dtype == torch.float64

    def test_soft_no_constraints_zero_energy(self, vc):
        """With no constraints, soft energy = 0."""
        x_t = _make_all_mask_tokens(vc, 4)
        pad_mask = vc.compute_pad_mask(4)
        node_logits = torch.randn(vc.n_max, NODE_VOCAB_SIZE, dtype=torch.float64)
        edge_logits = torch.randn(vc.n_edges, EDGE_VOCAB_SIZE, dtype=torch.float64)
        node_probs, edge_probs = build_effective_probs(
            x_t, node_logits, edge_logits, pad_mask, vc
        )

        composer = RewardComposer(constraints=[], phi="linear")
        energy, _ = composer.compute_energy_soft(
            node_probs, edge_probs, pad_mask, vc
        )
        assert energy.item() == 0.0
        assert energy.dtype == torch.float64
