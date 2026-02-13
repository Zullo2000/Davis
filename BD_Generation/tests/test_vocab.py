"""Tests for bd_gen.data.vocab module."""

import pytest
import torch

from bd_gen.data.vocab import (
    EDGE_MASK_IDX,
    EDGE_NO_EDGE_IDX,
    EDGE_PAD_IDX,
    EDGE_TYPES,
    EDGE_VOCAB_SIZE,
    NODE_MASK_IDX,
    NODE_PAD_IDX,
    NODE_TYPES,
    NODE_VOCAB_SIZE,
    RESPLAN_VOCAB_CONFIG,
    RPLAN_VOCAB_CONFIG,
    VocabConfig,
)


class TestVocabConstants:
    """Verify vocabulary sizes and index assignments."""

    def test_node_type_count(self):
        assert len(NODE_TYPES) == 13

    def test_node_vocab_size(self):
        assert NODE_VOCAB_SIZE == len(NODE_TYPES) + 2  # +MASK +PAD

    def test_node_special_indices(self):
        assert NODE_MASK_IDX == 13
        assert NODE_PAD_IDX == 14
        assert NODE_MASK_IDX == len(NODE_TYPES)
        assert NODE_PAD_IDX == len(NODE_TYPES) + 1

    def test_edge_type_count(self):
        assert len(EDGE_TYPES) == 10

    def test_edge_vocab_size(self):
        assert EDGE_VOCAB_SIZE == len(EDGE_TYPES) + 3  # +no-edge +MASK +PAD

    def test_edge_special_indices(self):
        assert EDGE_NO_EDGE_IDX == 10
        assert EDGE_MASK_IDX == 11
        assert EDGE_PAD_IDX == 12
        assert EDGE_NO_EDGE_IDX == len(EDGE_TYPES)

    def test_no_duplicate_node_types(self):
        assert len(NODE_TYPES) == len(set(NODE_TYPES))

    def test_no_duplicate_edge_types(self):
        assert len(EDGE_TYPES) == len(set(EDGE_TYPES))


class TestVocabConfigProperties:
    """Verify VocabConfig derived properties for multiple n_max values."""

    @pytest.mark.parametrize(
        "n_max,expected_n_edges,expected_seq_len",
        [
            (2, 1, 3),
            (3, 3, 6),
            (8, 28, 36),
            (14, 91, 105),
        ],
    )
    def test_n_edges_and_seq_len(self, n_max, expected_n_edges, expected_seq_len):
        vc = VocabConfig(n_max=n_max)
        assert vc.n_edges == expected_n_edges
        assert vc.seq_len == expected_seq_len

    def test_rplan_preset(self):
        assert RPLAN_VOCAB_CONFIG.n_max == 8
        assert RPLAN_VOCAB_CONFIG.seq_len == 36

    def test_resplan_preset(self):
        assert RESPLAN_VOCAB_CONFIG.n_max == 14
        assert RESPLAN_VOCAB_CONFIG.seq_len == 105

    def test_frozen(self):
        with pytest.raises(AttributeError):
            RPLAN_VOCAB_CONFIG.n_max = 10  # type: ignore[misc]


class TestEdgePositionBijectivity:
    """Verify edge_position_to_pair and pair_to_edge_position are inverses."""

    @pytest.mark.parametrize("n_max", [2, 3, 5, 8, 14])
    def test_roundtrip_pos_to_pair_to_pos(self, n_max):
        vc = VocabConfig(n_max=n_max)
        for pos in range(vc.n_edges):
            i, j = vc.edge_position_to_pair(pos)
            assert 0 <= i < j < n_max
            assert vc.pair_to_edge_position(i, j) == pos

    @pytest.mark.parametrize("n_max", [2, 3, 5, 8, 14])
    def test_roundtrip_pair_to_pos_to_pair(self, n_max):
        vc = VocabConfig(n_max=n_max)
        for i in range(n_max):
            for j in range(i + 1, n_max):
                pos = vc.pair_to_edge_position(i, j)
                assert 0 <= pos < vc.n_edges
                assert vc.edge_position_to_pair(pos) == (i, j)

    @pytest.mark.parametrize("n_max", [2, 3, 5, 8, 14])
    def test_all_positions_unique(self, n_max):
        vc = VocabConfig(n_max=n_max)
        pairs = [vc.edge_position_to_pair(p) for p in range(vc.n_edges)]
        assert len(pairs) == len(set(pairs))

    def test_swapped_pair_gives_same_position(self):
        vc = RPLAN_VOCAB_CONFIG
        assert vc.pair_to_edge_position(2, 5) == vc.pair_to_edge_position(5, 2)

    def test_self_loop_raises(self):
        vc = RPLAN_VOCAB_CONFIG
        with pytest.raises(ValueError):
            vc.pair_to_edge_position(3, 3)

    def test_out_of_range_raises(self):
        vc = RPLAN_VOCAB_CONFIG
        with pytest.raises(ValueError):
            vc.edge_position_to_pair(28)
        with pytest.raises(ValueError):
            vc.edge_position_to_pair(-1)
        with pytest.raises(ValueError):
            vc.pair_to_edge_position(0, 8)


class TestComputePadMask:
    """Verify compute_pad_mask produces correct masks."""

    def test_full_occupancy_no_pad(self):
        vc = RPLAN_VOCAB_CONFIG  # n_max=8
        mask = vc.compute_pad_mask(num_rooms=8)
        assert mask.shape == (36,)
        assert mask.dtype == torch.bool
        assert mask.all()  # no PAD when all rooms used

    def test_single_room(self):
        vc = RPLAN_VOCAB_CONFIG
        mask = vc.compute_pad_mask(num_rooms=1)
        # Only position 0 (node 0) is real; 0 real edges (need 2 nodes for an edge)
        assert mask[0].item() is True
        assert mask[1:8].sum() == 0  # remaining nodes are PAD
        assert mask[8:].sum() == 0  # all edges are PAD

    def test_two_rooms(self):
        vc = RPLAN_VOCAB_CONFIG
        mask = vc.compute_pad_mask(num_rooms=2)
        assert mask[:2].all()  # 2 real nodes
        assert mask[2:8].sum() == 0  # 6 PAD nodes
        # Edge (0,1) at position 0 is real
        assert mask[8].item() is True
        assert mask[9:].sum() == 0  # remaining edges are PAD

    @pytest.mark.parametrize("num_rooms", range(1, 9))
    def test_real_node_count(self, num_rooms):
        vc = RPLAN_VOCAB_CONFIG
        mask = vc.compute_pad_mask(num_rooms)
        assert mask[:8].sum() == num_rooms

    @pytest.mark.parametrize("num_rooms", range(1, 9))
    def test_real_edge_count(self, num_rooms):
        vc = RPLAN_VOCAB_CONFIG
        mask = vc.compute_pad_mask(num_rooms)
        expected_real_edges = num_rooms * (num_rooms - 1) // 2
        assert mask[8:].sum() == expected_real_edges

    def test_invalid_num_rooms_zero(self):
        with pytest.raises(ValueError):
            RPLAN_VOCAB_CONFIG.compute_pad_mask(0)

    def test_invalid_num_rooms_exceeds_max(self):
        with pytest.raises(ValueError):
            RPLAN_VOCAB_CONFIG.compute_pad_mask(9)
