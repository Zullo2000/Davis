"""Tests for bd_gen.data.graph2plan_loader module."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from bd_gen.data.graph2plan_loader import (
    _invert_relationship,
    _parse_record,
    load_graph2plan,
)
from bd_gen.data.vocab import EDGE_TYPES, NODE_TYPES

# ---------------------------------------------------------------------------
# Path to real data.mat (skip tests if not present)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MAT_PATH = _PROJECT_ROOT / "data" / "data.mat"
_CACHE_PATH = _PROJECT_ROOT / "data_cache" / "graph2plan_nmax8_test.pt"

needs_real_data = pytest.mark.skipif(
    not _MAT_PATH.exists(),
    reason=f"Real data.mat not found at {_MAT_PATH}",
)


# ===================================================================
# TestRelationshipInversion
# ===================================================================


class TestRelationshipInversion:
    """Verify _invert_relationship utility."""

    def test_specific_pairs(self):
        """Check all 5 documented inverse pairs."""
        assert _invert_relationship(0) == 9  # left-above <-> right-below
        assert _invert_relationship(1) == 8  # left-below <-> right-above
        assert _invert_relationship(2) == 7  # left-of <-> right-of
        assert _invert_relationship(3) == 6  # above <-> below
        assert _invert_relationship(4) == 5  # inside <-> surrounding

    def test_roundtrip(self):
        """Inverting twice returns the original."""
        for r in range(10):
            assert _invert_relationship(_invert_relationship(r)) == r

    def test_symmetry_property(self):
        """Inverse is 9 - r for all valid indices."""
        for r in range(10):
            assert _invert_relationship(r) == 9 - r

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            _invert_relationship(-1)
        with pytest.raises(ValueError):
            _invert_relationship(10)
        with pytest.raises(ValueError):
            _invert_relationship(100)


# ===================================================================
# TestParseSynthetic
# ===================================================================


def _make_record(
    rtype: list[int],
    redge: list[list[int]] | None = None,
) -> SimpleNamespace:
    """Create a mock Graph2Plan record with the given data.

    Args:
        rtype: List of room type indices.
        redge: List of [u, v, rel] triples, or None for empty edges.

    Returns:
        SimpleNamespace mimicking a scipy struct record.
    """
    ns = SimpleNamespace()
    ns.rType = np.array(rtype, dtype=np.int32)

    if redge is None or len(redge) == 0:
        ns.rEdge = np.array([], dtype=np.int32).reshape(0, 3)
    else:
        ns.rEdge = np.array(redge, dtype=np.int32)

    return ns


class TestParseSynthetic:
    """Test graph parsing logic with synthetic mock records."""

    def test_basic_graph_keys(self):
        """Parsed dict has the required keys."""
        rec = _make_record(
            rtype=[0, 1, 3],
            redge=[[0, 1, 2], [0, 2, 3], [1, 2, 6]],
        )
        result = _parse_record(rec)
        assert result is not None
        assert set(result.keys()) == {"node_types", "edge_triples", "num_rooms"}

    def test_basic_graph_values(self):
        """Parsed dict has correct values."""
        rec = _make_record(
            rtype=[0, 1, 3],
            redge=[[0, 1, 2], [0, 2, 3], [1, 2, 6]],
        )
        result = _parse_record(rec)
        assert result is not None
        assert result["node_types"] == [0, 1, 3]
        assert result["num_rooms"] == 3
        assert len(result["edge_triples"]) == 3
        assert (0, 1, 2) in result["edge_triples"]
        assert (0, 2, 3) in result["edge_triples"]
        assert (1, 2, 6) in result["edge_triples"]

    def test_self_loop_filtering(self):
        """Self-loops (u == v) are filtered out."""
        rec = _make_record(
            rtype=[0, 1, 3],
            redge=[[0, 0, 4], [0, 1, 2], [2, 2, 5]],
        )
        result = _parse_record(rec)
        assert result is not None
        assert len(result["edge_triples"]) == 1
        assert result["edge_triples"][0] == (0, 1, 2)

    def test_single_room_scalar_rtype(self):
        """Single room: rType might be scalar after squeeze_me=True."""
        ns = SimpleNamespace()
        # Simulate scalar rType (what squeeze_me=True produces for a 1-element array)
        ns.rType = np.int32(5)
        ns.rEdge = np.array([], dtype=np.int32).reshape(0, 3)

        result = _parse_record(ns)
        assert result is not None
        assert result["node_types"] == [5]
        assert result["num_rooms"] == 1
        assert result["edge_triples"] == []

    def test_empty_edges(self):
        """Graph with no edges parses correctly."""
        rec = _make_record(rtype=[0, 1], redge=[])
        result = _parse_record(rec)
        assert result is not None
        assert result["edge_triples"] == []
        assert result["num_rooms"] == 2

    def test_single_edge_1d_shape(self):
        """A single edge might be squeezed to 1-D by loadmat."""
        ns = SimpleNamespace()
        ns.rType = np.array([0, 1], dtype=np.int32)
        # Simulate squeeze_me reducing a (1,3) array to (3,)
        ns.rEdge = np.array([0, 1, 2], dtype=np.int32)

        result = _parse_record(ns)
        assert result is not None
        assert len(result["edge_triples"]) == 1
        assert result["edge_triples"][0] == (0, 1, 2)

    def test_lower_triangle_edge_inverted(self):
        """Edges with u > v are swapped and relationship inverted."""
        rec = _make_record(
            rtype=[0, 1, 3],
            # Edge (2, 0, 3) has u > v, should become (0, 2, 6) after inversion
            redge=[[2, 0, 3]],
        )
        result = _parse_record(rec)
        assert result is not None
        assert len(result["edge_triples"]) == 1
        assert result["edge_triples"][0] == (0, 2, 6)  # 9 - 3 = 6


class TestLoadGraph2PlanSynthetic:
    """Test load_graph2plan with mocked .mat data."""

    def test_n_max_filtering(self, tmp_path: Path):
        """Graphs exceeding n_max are filtered out."""
        # Create synthetic .mat-like data with varying room counts
        records = np.array(
            [
                _make_record(rtype=[0, 1], redge=[[0, 1, 2]]),
                _make_record(rtype=[0, 1, 2], redge=[[0, 1, 3], [1, 2, 5]]),
                # 4 rooms -- will be filtered with n_max=3
                _make_record(
                    rtype=[0, 1, 2, 3],
                    redge=[[0, 1, 2], [0, 2, 3], [0, 3, 4]],
                ),
            ],
            dtype=object,
        )

        mat_path = tmp_path / "data.mat"
        mat_path.touch()  # Create dummy file so existence check passes
        cache_path = tmp_path / "cache.pt"

        with patch("bd_gen.data.graph2plan_loader.sio.loadmat") as mock_load:
            mock_load.return_value = {"data": records}
            result = load_graph2plan(mat_path, cache_path, n_max=3)

        assert len(result) == 2
        assert all(g["num_rooms"] <= 3 for g in result)

    def test_cache_roundtrip(self, tmp_path: Path):
        """Cached result matches original parse."""
        records = np.array(
            [
                _make_record(rtype=[0, 1, 3], redge=[[0, 1, 2], [0, 2, 3]]),
                _make_record(rtype=[0, 1], redge=[[0, 1, 7]]),
            ],
            dtype=object,
        )

        mat_path = tmp_path / "data.mat"
        mat_path.touch()  # Create dummy file so existence check passes
        cache_path = tmp_path / "cache.pt"

        with patch("bd_gen.data.graph2plan_loader.sio.loadmat") as mock_load:
            mock_load.return_value = {"data": records}
            original = load_graph2plan(mat_path, cache_path, n_max=8)

        # Load from cache (no mock needed)
        cached = load_graph2plan(mat_path, cache_path, n_max=8)

        assert len(cached) == len(original)
        for orig, cach in zip(original, cached):
            assert orig["node_types"] == cach["node_types"]
            assert orig["edge_triples"] == cach["edge_triples"]
            assert orig["num_rooms"] == cach["num_rooms"]

    def test_missing_mat_raises(self, tmp_path: Path):
        """FileNotFoundError when .mat is missing and no cache."""
        mat_path = tmp_path / "nonexistent.mat"
        cache_path = tmp_path / "cache.pt"

        with pytest.raises(FileNotFoundError):
            load_graph2plan(mat_path, cache_path)


# ===================================================================
# TestParseReal (requires actual data.mat)
# ===================================================================


@needs_real_data
class TestParseReal:
    """Integration tests against the real Graph2Plan data.mat.

    These tests are skipped when data.mat is not present on the
    filesystem (e.g. in CI without data download).
    """

    @pytest.fixture(scope="class")
    def graphs(self) -> list[dict]:
        """Parse data.mat once and share across all tests in this class."""
        # Use a test-specific cache to avoid polluting the real cache
        return load_graph2plan(_MAT_PATH, _CACHE_PATH, n_max=8)

    def test_non_empty(self, graphs: list[dict]):
        assert len(graphs) > 0

    def test_count_reasonable(self, graphs: list[dict]):
        """Expect > 50K graphs (dataset has ~80K records)."""
        assert len(graphs) > 50_000

    def test_all_node_types_valid(self, graphs: list[dict]):
        """All node types are in [0, 12]."""
        for g in graphs:
            for nt in g["node_types"]:
                assert 0 <= nt < len(NODE_TYPES), f"Invalid node type {nt}"

    def test_all_edge_types_valid(self, graphs: list[dict]):
        """All edge relationship types are in [0, 9]."""
        for g in graphs:
            for _u, _v, rel in g["edge_triples"]:
                assert 0 <= rel < len(EDGE_TYPES), f"Invalid edge type {rel}"

    def test_all_edges_upper_triangle(self, graphs: list[dict]):
        """All edges satisfy u < v (strict upper triangle)."""
        for g in graphs:
            for u, v, _rel in g["edge_triples"]:
                assert u < v, f"Edge ({u}, {v}) is not upper-triangle"

    def test_num_rooms_matches_node_types(self, graphs: list[dict]):
        for g in graphs:
            assert g["num_rooms"] == len(g["node_types"])

    def test_no_graphs_exceed_nmax(self, graphs: list[dict]):
        for g in graphs:
            assert g["num_rooms"] <= 8

    def test_cache_roundtrip(self, graphs: list[dict]):
        """Reloading from cache gives identical results."""
        reloaded = torch.load(_CACHE_PATH, weights_only=False)
        assert len(reloaded) == len(graphs)
        # Spot-check first and last records
        for idx in [0, -1, len(graphs) // 2]:
            assert reloaded[idx]["node_types"] == graphs[idx]["node_types"]
            assert reloaded[idx]["edge_triples"] == graphs[idx]["edge_triples"]
            assert reloaded[idx]["num_rooms"] == graphs[idx]["num_rooms"]

    def test_room_count_distribution(self, graphs: list[dict]):
        """Room counts are within expected range (data has 4-8 rooms)."""
        room_counts = [g["num_rooms"] for g in graphs]
        assert min(room_counts) >= 1  # code should handle 1, data has >= 4
        assert max(room_counts) <= 8
