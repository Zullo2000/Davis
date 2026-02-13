"""Tests for bd_gen.data.tokenizer module.

Covers tokenize/detokenize correctness, round-trip identity, and the
critical PAD-vs-no-edge invariant across all supported graph sizes.
"""

import pytest
import torch

from bd_gen.data.tokenizer import detokenize, tokenize
from bd_gen.data.vocab import (
    EDGE_NO_EDGE_IDX,
    EDGE_PAD_IDX,
    EDGE_TYPES,
    NODE_PAD_IDX,
    NODE_TYPES,
    VocabConfig,
)

# ---------------------------------------------------------------------------
# Helpers for programmatic graph generation
# ---------------------------------------------------------------------------


def make_graph(
    num_rooms: int,
    node_types: list[int] | None = None,
    edge_triples: list[tuple[int, int, int]] | None = None,
) -> dict:
    """Build a graph dict for testing.

    Args:
        num_rooms: Number of rooms.
        node_types: Room type indices. If None, cycles through valid types.
        edge_triples: Edge list. If None, defaults to empty.

    Returns:
        Graph dict with ``num_rooms``, ``node_types``, ``edge_triples``.
    """
    if node_types is None:
        node_types = [i % len(NODE_TYPES) for i in range(num_rooms)]
    if edge_triples is None:
        edge_triples = []
    return {
        "num_rooms": num_rooms,
        "node_types": node_types,
        "edge_triples": edge_triples,
    }


def make_fully_connected_graph(
    num_rooms: int,
    node_types: list[int] | None = None,
) -> dict:
    """Build a graph where every pair of rooms has an edge.

    Each edge ``(i, j)`` gets relationship ``(i + j) % len(EDGE_TYPES)``
    for deterministic variety.
    """
    if node_types is None:
        node_types = [i % len(NODE_TYPES) for i in range(num_rooms)]
    edge_triples = []
    for i in range(num_rooms):
        for j in range(i + 1, num_rooms):
            rel = (i + j) % len(EDGE_TYPES)
            edge_triples.append((i, j, rel))
    return {
        "num_rooms": num_rooms,
        "node_types": node_types,
        "edge_triples": edge_triples,
    }


def make_random_graph(num_rooms: int, seed: int = 0) -> dict:
    """Build a random valid graph for round-trip testing.

    Uses a fixed seed for reproducibility. Each possible edge has a 50%
    chance of being present with a random relationship type.
    """
    rng = torch.Generator().manual_seed(seed)
    node_types = [
        int(torch.randint(0, len(NODE_TYPES), (1,), generator=rng).item())
        for _ in range(num_rooms)
    ]
    edge_triples = []
    for i in range(num_rooms):
        for j in range(i + 1, num_rooms):
            if torch.rand(1, generator=rng).item() < 0.5:
                rel = int(
                    torch.randint(0, len(EDGE_TYPES), (1,), generator=rng).item()
                )
                edge_triples.append((i, j, rel))
    return {
        "num_rooms": num_rooms,
        "node_types": node_types,
        "edge_triples": edge_triples,
    }


# ---------------------------------------------------------------------------
# TestTokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    """Verify the tokenize() function produces correct outputs."""

    def test_output_shapes(self, vocab_config: VocabConfig):
        """Tokens and pad_mask have shape (seq_len,)."""
        graph = make_graph(num_rooms=4)
        tokens, pad_mask = tokenize(graph, vocab_config)
        assert tokens.shape == (vocab_config.seq_len,)
        assert pad_mask.shape == (vocab_config.seq_len,)

    def test_output_dtypes(self, vocab_config: VocabConfig):
        """Tokens are long, pad_mask is bool."""
        graph = make_graph(num_rooms=4)
        tokens, pad_mask = tokenize(graph, vocab_config)
        assert tokens.dtype == torch.long
        assert pad_mask.dtype == torch.bool

    def test_pad_positions_have_pad_tokens(self, vocab_config: VocabConfig):
        """Where pad_mask is False, tokens must be NODE_PAD_IDX or EDGE_PAD_IDX."""
        graph = make_graph(num_rooms=3, edge_triples=[(0, 1, 2)])
        tokens, pad_mask = tokenize(graph, vocab_config)

        for idx in range(vocab_config.seq_len):
            if not pad_mask[idx]:
                val = int(tokens[idx].item())
                if idx < vocab_config.n_max:
                    assert val == NODE_PAD_IDX, (
                        f"PAD node position {idx} has value {val}, "
                        f"expected NODE_PAD_IDX={NODE_PAD_IDX}"
                    )
                else:
                    assert val == EDGE_PAD_IDX, (
                        f"PAD edge position {idx} has value {val}, "
                        f"expected EDGE_PAD_IDX={EDGE_PAD_IDX}"
                    )

    def test_no_edge_vs_pad_distinction(self, vocab_config: VocabConfig):
        """3-room graph with 1 edge: real edge positions get NO_EDGE."""
        # 3 rooms -> 3 real edge positions: (0,1), (0,2), (1,2)
        # Only provide edge (0, 1, 3) -> positions (0,2) and (1,2) should be NO_EDGE
        graph = make_graph(
            num_rooms=3,
            node_types=[0, 1, 2],
            edge_triples=[(0, 1, 3)],
        )
        tokens, pad_mask = tokenize(graph, vocab_config)

        # Collect real edge positions and their token values
        real_edge_values = {}
        for pos in range(vocab_config.n_edges):
            seq_idx = vocab_config.n_max + pos
            if pad_mask[seq_idx]:
                real_edge_values[pos] = int(tokens[seq_idx].item())

        # Should have 3 real edge positions for 3 rooms
        assert len(real_edge_values) == 3

        # Position for (0, 1) should have rel=3
        pos_01 = vocab_config.pair_to_edge_position(0, 1)
        assert real_edge_values[pos_01] == 3

        # Positions for (0, 2) and (1, 2) should have EDGE_NO_EDGE_IDX
        pos_02 = vocab_config.pair_to_edge_position(0, 2)
        pos_12 = vocab_config.pair_to_edge_position(1, 2)
        assert real_edge_values[pos_02] == EDGE_NO_EDGE_IDX
        assert real_edge_values[pos_12] == EDGE_NO_EDGE_IDX

        # Remaining edge positions (involving rooms >= 3) should be EDGE_PAD_IDX
        for pos in range(vocab_config.n_edges):
            seq_idx = vocab_config.n_max + pos
            if not pad_mask[seq_idx]:
                assert int(tokens[seq_idx].item()) == EDGE_PAD_IDX

    def test_full_graph_no_pad(self, vocab_config: VocabConfig):
        """n_max-room graph has an all-True pad_mask."""
        graph = make_fully_connected_graph(num_rooms=vocab_config.n_max)
        tokens, pad_mask = tokenize(graph, vocab_config)
        assert pad_mask.all(), "Full-occupancy graph should have no PAD positions"

    def test_single_room_graph(self, vocab_config: VocabConfig):
        """1 room, 0 real edges, everything else PAD."""
        graph = make_graph(num_rooms=1, node_types=[5], edge_triples=[])
        tokens, pad_mask = tokenize(graph, vocab_config)

        # Only position 0 is real
        assert pad_mask[0].item() is True
        assert pad_mask[1:].sum() == 0

        # Token at position 0 is the node type
        assert int(tokens[0].item()) == 5

        # All other positions are PAD
        for idx in range(1, vocab_config.n_max):
            assert int(tokens[idx].item()) == NODE_PAD_IDX
        for idx in range(vocab_config.n_max, vocab_config.seq_len):
            assert int(tokens[idx].item()) == EDGE_PAD_IDX

    def test_node_values_in_valid_range(self, vocab_config: VocabConfig):
        """Real node positions have values in [0, 12]."""
        for num_rooms in range(1, vocab_config.n_max + 1):
            graph = make_graph(num_rooms=num_rooms)
            tokens, pad_mask = tokenize(graph, vocab_config)
            for k in range(num_rooms):
                val = int(tokens[k].item())
                assert 0 <= val < len(NODE_TYPES), (
                    f"Node {k} value {val} out of range [0, {len(NODE_TYPES)})"
                )

    def test_edge_values_in_valid_range(self, vocab_config: VocabConfig):
        """Real edge positions have values in [0, EDGE_NO_EDGE_IDX]."""
        graph = make_fully_connected_graph(num_rooms=5)
        tokens, pad_mask = tokenize(graph, vocab_config)
        for pos in range(vocab_config.n_edges):
            seq_idx = vocab_config.n_max + pos
            if pad_mask[seq_idx]:
                val = int(tokens[seq_idx].item())
                assert 0 <= val <= EDGE_NO_EDGE_IDX, (
                    f"Real edge position {pos} value {val} out of range "
                    f"[0, {EDGE_NO_EDGE_IDX}]"
                )

    def test_invalid_num_rooms_zero(self, vocab_config: VocabConfig):
        """num_rooms=0 raises ValueError."""
        graph = make_graph(num_rooms=0, node_types=[])
        with pytest.raises(ValueError, match="num_rooms"):
            tokenize(graph, vocab_config)

    def test_invalid_num_rooms_exceeds_max(self, vocab_config: VocabConfig):
        """num_rooms > n_max raises ValueError."""
        n = vocab_config.n_max + 1
        graph = make_graph(num_rooms=n, node_types=list(range(n)))
        with pytest.raises(ValueError, match="num_rooms"):
            tokenize(graph, vocab_config)

    def test_invalid_node_type(self, vocab_config: VocabConfig):
        """Out-of-range node type raises ValueError."""
        graph = make_graph(num_rooms=2, node_types=[0, 99])
        with pytest.raises(ValueError, match="node_types"):
            tokenize(graph, vocab_config)

    def test_invalid_edge_triple_order(self, vocab_config: VocabConfig):
        """Edge triple with i >= j raises ValueError."""
        graph = make_graph(
            num_rooms=3,
            edge_triples=[(2, 1, 0)],  # i > j
        )
        with pytest.raises(ValueError, match="i < j"):
            tokenize(graph, vocab_config)

    def test_invalid_edge_relation(self, vocab_config: VocabConfig):
        """Out-of-range edge relationship raises ValueError."""
        graph = make_graph(
            num_rooms=3,
            edge_triples=[(0, 1, 99)],
        )
        with pytest.raises(ValueError, match="rel"):
            tokenize(graph, vocab_config)

    def test_mismatched_node_types_length(self, vocab_config: VocabConfig):
        """node_types length != num_rooms raises ValueError."""
        graph = {
            "num_rooms": 3,
            "node_types": [0, 1],  # length 2, not 3
            "edge_triples": [],
        }
        with pytest.raises(ValueError, match="node_types"):
            tokenize(graph, vocab_config)


# ---------------------------------------------------------------------------
# TestDetokenize
# ---------------------------------------------------------------------------


class TestDetokenize:
    """Verify the detokenize() function and round-trip identity."""

    def test_roundtrip_identity(self, vocab_config: VocabConfig):
        """tokenize then detokenize recovers the original graph."""
        graph = make_graph(
            num_rooms=4,
            node_types=[0, 3, 7, 12],
            edge_triples=[(0, 1, 2), (0, 3, 5), (1, 2, 9), (2, 3, 0)],
        )
        tokens, pad_mask = tokenize(graph, vocab_config)
        recovered = detokenize(tokens, pad_mask, vocab_config)

        assert recovered["num_rooms"] == graph["num_rooms"]
        assert recovered["node_types"] == graph["node_types"]
        assert sorted(recovered["edge_triples"]) == sorted(graph["edge_triples"])

    def test_roundtrip_single_room(self, vocab_config: VocabConfig):
        """1-room graph round-trip."""
        graph = make_graph(num_rooms=1, node_types=[10], edge_triples=[])
        tokens, pad_mask = tokenize(graph, vocab_config)
        recovered = detokenize(tokens, pad_mask, vocab_config)

        assert recovered["num_rooms"] == 1
        assert recovered["node_types"] == [10]
        assert recovered["edge_triples"] == []

    def test_roundtrip_max_rooms(self, vocab_config: VocabConfig):
        """8-room fully connected graph round-trip."""
        graph = make_fully_connected_graph(num_rooms=vocab_config.n_max)
        tokens, pad_mask = tokenize(graph, vocab_config)
        recovered = detokenize(tokens, pad_mask, vocab_config)

        assert recovered["num_rooms"] == vocab_config.n_max
        assert recovered["node_types"] == graph["node_types"]
        assert sorted(recovered["edge_triples"]) == sorted(graph["edge_triples"])

    @pytest.mark.parametrize("num_rooms", range(1, 9))
    def test_roundtrip_varying_sizes(
        self, num_rooms: int, vocab_config: VocabConfig
    ):
        """Parametrized round-trip for num_rooms 1-8 with random graphs."""
        graph = make_random_graph(num_rooms, seed=num_rooms * 42)
        tokens, pad_mask = tokenize(graph, vocab_config)
        recovered = detokenize(tokens, pad_mask, vocab_config)

        assert recovered["num_rooms"] == graph["num_rooms"]
        assert recovered["node_types"] == graph["node_types"]
        assert sorted(recovered["edge_triples"]) == sorted(graph["edge_triples"])

    def test_no_edge_not_in_triples(self, vocab_config: VocabConfig):
        """detokenize excludes no-edge positions from edge_triples."""
        # 4-room graph with only 1 edge -> 5 other real positions are no-edge
        graph = make_graph(
            num_rooms=4,
            edge_triples=[(1, 3, 7)],
        )
        tokens, pad_mask = tokenize(graph, vocab_config)
        recovered = detokenize(tokens, pad_mask, vocab_config)

        # Should only recover the one edge we put in
        assert len(recovered["edge_triples"]) == 1
        assert recovered["edge_triples"][0] == (1, 3, 7)

    def test_empty_edge_triples(self, vocab_config: VocabConfig):
        """Graph with no edges round-trips correctly."""
        graph = make_graph(num_rooms=5, edge_triples=[])
        tokens, pad_mask = tokenize(graph, vocab_config)
        recovered = detokenize(tokens, pad_mask, vocab_config)

        assert recovered["num_rooms"] == 5
        assert recovered["edge_triples"] == []

    def test_roundtrip_all_edge_types(self, vocab_config: VocabConfig):
        """Each of the 10 edge relationship types survives a round-trip."""
        # Use a graph with enough rooms to have 10 edges: 5 rooms -> C(5,2) = 10
        num_rooms = 5
        node_types = list(range(num_rooms))
        edge_triples = []
        rel_idx = 0
        for i in range(num_rooms):
            for j in range(i + 1, num_rooms):
                edge_triples.append((i, j, rel_idx % len(EDGE_TYPES)))
                rel_idx += 1

        graph = make_graph(
            num_rooms=num_rooms,
            node_types=node_types,
            edge_triples=edge_triples,
        )
        tokens, pad_mask = tokenize(graph, vocab_config)
        recovered = detokenize(tokens, pad_mask, vocab_config)

        assert sorted(recovered["edge_triples"]) == sorted(graph["edge_triples"])

    def test_invalid_tokens_shape(self, vocab_config: VocabConfig):
        """Wrong token shape raises ValueError."""
        tokens = torch.zeros(10, dtype=torch.long)
        pad_mask = torch.ones(10, dtype=torch.bool)
        with pytest.raises(ValueError, match="shape"):
            detokenize(tokens, pad_mask, vocab_config)

    def test_invalid_pad_mask_shape(self, vocab_config: VocabConfig):
        """Wrong pad_mask shape raises ValueError."""
        tokens = torch.zeros(vocab_config.seq_len, dtype=torch.long)
        pad_mask = torch.ones(10, dtype=torch.bool)
        with pytest.raises(ValueError, match="shape"):
            detokenize(tokens, pad_mask, vocab_config)

    def test_invalid_node_value_in_detokenize(self, vocab_config: VocabConfig):
        """Out-of-range node value in tokens raises ValueError."""
        graph = make_graph(num_rooms=2)
        tokens, pad_mask = tokenize(graph, vocab_config)
        tokens[0] = 99  # corrupt node value
        with pytest.raises(ValueError, match="node range"):
            detokenize(tokens, pad_mask, vocab_config)

    def test_invalid_edge_value_in_detokenize(self, vocab_config: VocabConfig):
        """Out-of-range edge relationship in tokens raises ValueError."""
        graph = make_graph(num_rooms=3, edge_triples=[(0, 1, 2)])
        tokens, pad_mask = tokenize(graph, vocab_config)
        # Find a real edge position and corrupt it with an invalid value
        pos_01 = vocab_config.pair_to_edge_position(0, 1)
        seq_idx = vocab_config.n_max + pos_01
        tokens[seq_idx] = 99  # not a valid rel and not EDGE_NO_EDGE_IDX
        with pytest.raises(ValueError, match="rel"):
            detokenize(tokens, pad_mask, vocab_config)


# ---------------------------------------------------------------------------
# TestPadCorrectness
# ---------------------------------------------------------------------------


class TestPadCorrectness:
    """Dedicated tests for the PAD invariant across all graph sizes."""

    @pytest.mark.parametrize("num_rooms", range(1, 9))
    def test_pad_mask_matches_compute_pad_mask(
        self, num_rooms: int, vocab_config: VocabConfig
    ):
        """Tokenize pad_mask matches VocabConfig.compute_pad_mask for all sizes."""
        graph = make_random_graph(num_rooms, seed=num_rooms)
        _, pad_mask = tokenize(graph, vocab_config)
        expected_mask = vocab_config.compute_pad_mask(num_rooms)
        assert torch.equal(pad_mask, expected_mask)

    @pytest.mark.parametrize("num_rooms", range(1, 9))
    def test_pad_never_contains_real_data(
        self, num_rooms: int, vocab_config: VocabConfig
    ):
        """PAD positions always have PAD tokens (never real data)."""
        graph = make_random_graph(num_rooms, seed=num_rooms + 100)
        tokens, pad_mask = tokenize(graph, vocab_config)

        for idx in range(vocab_config.seq_len):
            if not pad_mask[idx]:
                val = int(tokens[idx].item())
                if idx < vocab_config.n_max:
                    assert val == NODE_PAD_IDX, (
                        f"PAD node pos {idx}: expected {NODE_PAD_IDX}, got {val}"
                    )
                else:
                    assert val == EDGE_PAD_IDX, (
                        f"PAD edge pos {idx}: expected {EDGE_PAD_IDX}, got {val}"
                    )

    @pytest.mark.parametrize("num_rooms", range(1, 9))
    def test_real_never_contains_pad_token(
        self, num_rooms: int, vocab_config: VocabConfig
    ):
        """Real positions never have PAD tokens."""
        graph = make_random_graph(num_rooms, seed=num_rooms + 200)
        tokens, pad_mask = tokenize(graph, vocab_config)

        for idx in range(vocab_config.seq_len):
            if pad_mask[idx]:
                val = int(tokens[idx].item())
                if idx < vocab_config.n_max:
                    assert val != NODE_PAD_IDX, (
                        f"Real node pos {idx} has NODE_PAD_IDX"
                    )
                else:
                    assert val != EDGE_PAD_IDX, (
                        f"Real edge pos {idx} has EDGE_PAD_IDX"
                    )
