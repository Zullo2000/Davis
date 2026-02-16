"""Tests for bd_gen.eval.validity — graph validity checker."""

from __future__ import annotations

import torch

from bd_gen.data.tokenizer import tokenize
from bd_gen.data.vocab import (
    EDGE_MASK_IDX,
    NODE_MASK_IDX,
    RPLAN_VOCAB_CONFIG,
)
from bd_gen.eval.validity import check_validity, check_validity_batch

vc = RPLAN_VOCAB_CONFIG


# ---------------------------------------------------------------------------
# Helper: build tokens from a graph dict
# ---------------------------------------------------------------------------


def _make_tokens(graph_dict: dict):
    """Tokenize a graph dict and return (tokens, pad_mask)."""
    return tokenize(graph_dict, vc)


# ---------------------------------------------------------------------------
# Valid graphs
# ---------------------------------------------------------------------------


class TestValidGraphs:
    """Graphs that should pass all validity checks."""

    def test_valid_4room_connected(self):
        """A well-formed 4-room connected graph passes all checks."""
        graph = {
            "num_rooms": 4,
            "node_types": [0, 1, 2, 3],  # LivingRoom, MasterRoom, Kitchen, Bathroom
            "edge_triples": [(0, 1, 2), (1, 2, 3), (2, 3, 7)],
        }
        tokens, pad_mask = _make_tokens(graph)
        result = check_validity(tokens, pad_mask, vc)

        assert result["connected"] is True
        assert result["consistent"] is True
        assert result["valid_types"] is True
        assert result["no_mask_tokens"] is True
        assert result["no_out_of_range"] is True
        assert result["overall"] is True
        assert result["num_rooms"] == 4
        assert result["num_edges"] == 3

    def test_valid_single_room(self):
        """A 1-room graph is trivially valid (no edges needed)."""
        graph = {
            "num_rooms": 1,
            "node_types": [0],  # LivingRoom
            "edge_triples": [],
        }
        tokens, pad_mask = _make_tokens(graph)
        result = check_validity(tokens, pad_mask, vc)

        assert result["connected"] is True
        assert result["overall"] is True
        assert result["num_rooms"] == 1
        assert result["num_edges"] == 0

    def test_valid_2room(self):
        """A 2-room connected graph passes."""
        graph = {
            "num_rooms": 2,
            "node_types": [0, 7],  # LivingRoom, SecondRoom
            "edge_triples": [(0, 1, 2)],  # left-of
        }
        tokens, pad_mask = _make_tokens(graph)
        result = check_validity(tokens, pad_mask, vc)

        assert result["overall"] is True
        assert result["num_rooms"] == 2

    def test_valid_max_rooms(self):
        """An 8-room fully connected graph passes."""
        graph = {
            "num_rooms": 8,
            "node_types": [0, 1, 2, 3, 4, 5, 6, 7],
            "edge_triples": [
                (i, j, (i + j) % 10)
                for i in range(8) for j in range(i + 1, 8)
            ],
        }
        tokens, pad_mask = _make_tokens(graph)
        result = check_validity(tokens, pad_mask, vc)

        assert result["connected"] is True
        assert result["overall"] is True
        assert result["num_rooms"] == 8


# ---------------------------------------------------------------------------
# Connectivity failures
# ---------------------------------------------------------------------------


class TestConnectivity:
    """Graphs that fail the connectivity check."""

    def test_disconnected_graph(self):
        """4 rooms where rooms 2-3 are not reachable from 0."""
        graph = {
            "num_rooms": 4,
            "node_types": [0, 1, 2, 3],
            "edge_triples": [(0, 1, 2), (2, 3, 3)],  # {0,1} and {2,3} separate
        }
        tokens, pad_mask = _make_tokens(graph)
        result = check_validity(tokens, pad_mask, vc)

        assert result["connected"] is False
        assert result["overall"] is False

    def test_isolated_node(self):
        """3 rooms where room 2 has no edges at all."""
        graph = {
            "num_rooms": 3,
            "node_types": [0, 1, 2],
            "edge_triples": [(0, 1, 4)],  # Only 0-1 connected; 2 isolated
        }
        tokens, pad_mask = _make_tokens(graph)
        result = check_validity(tokens, pad_mask, vc)

        assert result["connected"] is False
        assert result["overall"] is False

    def test_no_edges_multi_room(self):
        """Multiple rooms with zero edges => disconnected."""
        graph = {
            "num_rooms": 3,
            "node_types": [0, 1, 2],
            "edge_triples": [],
        }
        tokens, pad_mask = _make_tokens(graph)
        result = check_validity(tokens, pad_mask, vc)

        assert result["connected"] is False


# ---------------------------------------------------------------------------
# MASK token detection
# ---------------------------------------------------------------------------


class TestMaskTokens:
    """Graphs with remaining MASK tokens should fail."""

    def test_node_mask_token(self):
        """Inject NODE_MASK_IDX into a real node position."""
        graph = {
            "num_rooms": 4,
            "node_types": [0, 1, 2, 3],
            "edge_triples": [(0, 1, 2), (1, 2, 3), (2, 3, 7)],
        }
        tokens, pad_mask = _make_tokens(graph)
        tokens[1] = NODE_MASK_IDX  # Corrupt node 1

        result = check_validity(tokens, pad_mask, vc)
        assert result["no_mask_tokens"] is False
        assert result["overall"] is False

    def test_edge_mask_token(self):
        """Inject EDGE_MASK_IDX into a real edge position."""
        graph = {
            "num_rooms": 4,
            "node_types": [0, 1, 2, 3],
            "edge_triples": [(0, 1, 2), (1, 2, 3), (2, 3, 7)],
        }
        tokens, pad_mask = _make_tokens(graph)
        # First real edge position
        first_real_edge = vc.n_max
        tokens[first_real_edge] = EDGE_MASK_IDX

        result = check_validity(tokens, pad_mask, vc)
        assert result["no_mask_tokens"] is False
        assert result["overall"] is False


# ---------------------------------------------------------------------------
# Out-of-range detection
# ---------------------------------------------------------------------------


class TestOutOfRange:
    """Tokens with values outside valid ranges."""

    def test_node_out_of_range(self):
        """Node token with value > 12 (but not MASK/PAD) flagged."""
        graph = {
            "num_rooms": 2,
            "node_types": [0, 1],
            "edge_triples": [(0, 1, 2)],
        }
        tokens, pad_mask = _make_tokens(graph)
        tokens[0] = 99  # Out of range

        result = check_validity(tokens, pad_mask, vc)
        assert result["no_out_of_range"] is False
        assert result["overall"] is False


# ---------------------------------------------------------------------------
# Room-type constraint violations
# ---------------------------------------------------------------------------


class TestRoomTypeConstraints:
    """Graphs violating room-type uniqueness constraints."""

    def test_duplicate_living_room(self):
        """Two LivingRoom (idx 0) nodes => valid_types=False."""
        graph = {
            "num_rooms": 3,
            "node_types": [0, 0, 2],  # Two LivingRooms
            "edge_triples": [(0, 1, 2), (1, 2, 3)],
        }
        tokens, pad_mask = _make_tokens(graph)
        result = check_validity(tokens, pad_mask, vc)

        assert result["valid_types"] is False
        assert result["overall"] is False
        assert "LivingRoom" in result["details"].get("type_violations", {})

    def test_duplicate_entrance(self):
        """Two Entrance (idx 10) nodes => valid_types=False."""
        graph = {
            "num_rooms": 3,
            "node_types": [0, 10, 10],  # Two Entrances
            "edge_triples": [(0, 1, 2), (1, 2, 3)],
        }
        tokens, pad_mask = _make_tokens(graph)
        result = check_validity(tokens, pad_mask, vc)

        assert result["valid_types"] is False
        assert "Entrance" in result["details"].get("type_violations", {})


# ---------------------------------------------------------------------------
# Overall flag consistency
# ---------------------------------------------------------------------------


class TestOverallFlag:
    """Verify overall is the AND of all individual checks."""

    def test_overall_is_and_of_checks(self):
        """Overall must be True only if all individual checks pass."""
        # Valid graph
        graph = {
            "num_rooms": 3,
            "node_types": [0, 1, 2],
            "edge_triples": [(0, 1, 2), (0, 2, 3), (1, 2, 7)],
        }
        tokens, pad_mask = _make_tokens(graph)
        result = check_validity(tokens, pad_mask, vc)

        expected_overall = (
            result["connected"]
            and result["consistent"]
            and result["valid_types"]
            and result["no_mask_tokens"]
            and result["no_out_of_range"]
        )
        assert result["overall"] == expected_overall

    def test_overall_false_when_disconnected_but_otherwise_valid(self):
        """Disconnected graph with valid types => overall=False."""
        graph = {
            "num_rooms": 4,
            "node_types": [0, 1, 2, 3],
            "edge_triples": [(0, 1, 2), (2, 3, 3)],
        }
        tokens, pad_mask = _make_tokens(graph)
        result = check_validity(tokens, pad_mask, vc)

        assert result["no_mask_tokens"] is True
        assert result["no_out_of_range"] is True
        assert result["connected"] is False
        assert result["overall"] is False


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------


class TestBatch:
    """Tests for check_validity_batch."""

    def test_batch_returns_correct_count(self):
        """Batch helper returns one result per sample."""
        graphs = [
            {
                "num_rooms": 2,
                "node_types": [0, 1],
                "edge_triples": [(0, 1, 2)],
            },
            {
                "num_rooms": 3,
                "node_types": [0, 1, 2],
                "edge_triples": [(0, 1, 2), (1, 2, 3)],
            },
        ]
        tokens_list = []
        masks_list = []
        for g in graphs:
            t, m = _make_tokens(g)
            tokens_list.append(t)
            masks_list.append(m)

        tokens = torch.stack(tokens_list)
        pad_mask = torch.stack(masks_list)

        results = check_validity_batch(tokens, pad_mask, vc)
        assert len(results) == 2

    def test_batch_matches_individual(self):
        """Batch results match individual calls."""
        graph = {
            "num_rooms": 3,
            "node_types": [0, 1, 2],
            "edge_triples": [(0, 1, 2), (0, 2, 3), (1, 2, 7)],
        }
        tokens, pad_mask = _make_tokens(graph)

        # Create a batch of 3 identical samples
        batch_tokens = tokens.unsqueeze(0).expand(3, -1).clone()
        batch_mask = pad_mask.unsqueeze(0).expand(3, -1).clone()

        batch_results = check_validity_batch(batch_tokens, batch_mask, vc)
        individual = check_validity(tokens, pad_mask, vc)

        for r in batch_results:
            assert r["overall"] == individual["overall"]
            assert r["connected"] == individual["connected"]
            assert r["num_rooms"] == individual["num_rooms"]
