"""Tests for the calibration protocol (spec tests 54-56).

Covers:
  - P90 computation on known violation distribution (test 54)
  - All-zero violations → P90 = 1.0 (test 55)
  - Save/load roundtrip produces identical dict (test 56)
  - Additional edge cases: single sample, all-same violations, mixed constraints
"""

from __future__ import annotations

import json

import pytest

from bd_gen.guidance.calibration import (
    calibrate_from_samples,
    load_calibration,
    save_calibration,
)
from bd_gen.guidance.constraints import ExactCount, ForbidAdj, RequireAdj


# =========================================================================
# Helpers
# =========================================================================


def _make_graph(
    node_types: list[int],
    edge_triples: list[tuple[int, int, int]] | None = None,
) -> dict:
    """Build a minimal graph_dict for constraint evaluation."""
    return {
        "num_rooms": len(node_types),
        "node_types": node_types,
        "edge_triples": edge_triples or [],
    }


# =========================================================================
# Test 54: P90 computation on known distribution
# =========================================================================


class TestP90Computation:
    """Spec test 54: known violation distribution → correct P90."""

    def test_p90_exact_count_known_distribution(self):
        """10 samples with violations [0, 0, 0, 1, 1, 1, 2, 2, 3, 5].
        Non-zero = [1, 1, 1, 2, 2, 3, 5], P90 = 90th percentile = 4.2.
        """
        # ExactCount(Kitchen=1): Kitchen is NODE_TYPES index 2
        constraint = ExactCount(name="one_kitchen", room_type_idx=2, target=1)

        # Build graphs producing known violation counts:
        # 1 kitchen → violation 0
        # 1 kitchen → violation 0
        # 1 kitchen → violation 0
        # 0 kitchens → violation 1
        # 2 kitchens → violation 1
        # 0 kitchens → violation 1
        # 3 kitchens → violation 2
        # 3 kitchens → violation 2
        # 4 kitchens → violation 3
        # 6 kitchens → violation 5
        graphs = [
            _make_graph([2]),              # 1 kitchen → v=0
            _make_graph([2, 0]),           # 1 kitchen → v=0
            _make_graph([2, 3]),           # 1 kitchen → v=0
            _make_graph([0, 3]),           # 0 kitchens → v=1
            _make_graph([2, 2]),           # 2 kitchens → v=1
            _make_graph([0, 0]),           # 0 kitchens → v=1
            _make_graph([2, 2, 2]),        # 3 kitchens → v=2
            _make_graph([2, 2, 2, 0]),     # 3 kitchens → v=2
            _make_graph([2, 2, 2, 2]),     # 4 kitchens → v=3
            _make_graph([2, 2, 2, 2, 2, 2]),  # 6 kitchens → v=5
        ]

        calibration = calibrate_from_samples(graphs, [constraint])

        # Non-zero violations: [1, 1, 1, 2, 2, 3, 5]
        # numpy.percentile([1, 1, 1, 2, 2, 3, 5], 90) = 3.8
        # (index = 0.90 * 6 = 5.4 → lerp(3, 5, 0.4) = 3.8)
        assert "one_kitchen" in calibration
        assert calibration["one_kitchen"] == pytest.approx(3.8, abs=0.01)

    def test_p90_forbid_adj_known_distribution(self):
        """ForbidAdj with graded violations."""
        # ForbidAdj(Bathroom-Kitchen): Bathroom=3, Kitchen=2
        constraint = ForbidAdj(name="no_bath_kitchen", type_a_idx=3, type_b_idx=2)

        graphs = [
            # No forbidden adj → v=0
            _make_graph([3, 2], []),
            _make_graph([3, 2], []),
            # 1 forbidden adj → v=1
            _make_graph([3, 2], [(0, 1, 1)]),
            _make_graph([3, 2], [(0, 1, 1)]),
            _make_graph([3, 2], [(0, 1, 1)]),
            # 2 forbidden adj → v=2
            _make_graph([3, 2, 3, 2], [(0, 1, 1), (2, 3, 1)]),
        ]

        calibration = calibrate_from_samples(graphs, [constraint])

        # Non-zero: [1, 1, 1, 2], P90 = numpy.percentile([1, 1, 1, 2], 90) = 1.7
        assert calibration["no_bath_kitchen"] == pytest.approx(1.7, abs=0.01)


# =========================================================================
# Test 55: All-zero violations → P90 = 1.0
# =========================================================================


class TestAllZeroViolations:
    """Spec test 55: constraint always satisfied → P90 = 1.0."""

    def test_all_satisfied_gives_p90_one(self):
        """When all samples satisfy the constraint, P90 defaults to 1.0."""
        constraint = ExactCount(name="one_kitchen", room_type_idx=2, target=1)

        # All graphs have exactly 1 kitchen
        graphs = [
            _make_graph([2, 0, 3]),
            _make_graph([2, 1]),
            _make_graph([2]),
            _make_graph([0, 2, 3, 4]),
        ]

        calibration = calibrate_from_samples(graphs, [constraint])
        assert calibration["one_kitchen"] == 1.0

    def test_require_adj_all_satisfied(self):
        """RequireAdj always satisfied → P90 = 1.0."""
        # RequireAdj(Kitchen-LivingRoom): Kitchen=2, LivingRoom=0
        constraint = RequireAdj(name="kitchen_near_living", type_a_idx=2, type_b_idx=0)

        graphs = [
            _make_graph([2, 0], [(0, 1, 1)]),
            _make_graph([0, 2], [(0, 1, 1)]),
            _make_graph([2, 0, 3], [(0, 1, 1)]),
        ]

        calibration = calibrate_from_samples(graphs, [constraint])
        assert calibration["kitchen_near_living"] == 1.0


# =========================================================================
# Test 56: Save/load roundtrip
# =========================================================================


class TestSaveLoadRoundtrip:
    """Spec test 56: save calibration JSON → reload → identical dict."""

    def test_roundtrip(self, tmp_path):
        """Save and reload calibration, verify exact match."""
        original = {
            "one_kitchen": 2.5,
            "one_living": 1.0,
            "kitchen_near_living": 1.0,
            "no_bath_kitchen": 1.7,
        }

        path = tmp_path / "calibration.json"
        save_calibration(path, original)

        # Verify file exists and is valid JSON
        assert path.exists()
        with open(path) as f:
            raw = json.load(f)
        assert raw == original

        # Reload via load_calibration
        loaded = load_calibration(path)
        assert loaded == original

    def test_roundtrip_nested_dir(self, tmp_path):
        """Save to a nested directory that doesn't exist yet."""
        original = {"test_constraint": 3.14}

        path = tmp_path / "nested" / "dir" / "calibration.json"
        save_calibration(path, original)

        loaded = load_calibration(path)
        assert loaded == original


# =========================================================================
# Additional edge cases
# =========================================================================


class TestCalibrationEdgeCases:

    def test_single_violated_sample(self):
        """Single sample with violation → P90 = that violation value."""
        constraint = ExactCount(name="one_kitchen", room_type_idx=2, target=1)
        graphs = [_make_graph([2, 2, 2])]  # 3 kitchens → v=2

        calibration = calibrate_from_samples(graphs, [constraint])
        assert calibration["one_kitchen"] == 2.0

    def test_multiple_constraints(self):
        """Calibrate multiple constraints simultaneously."""
        constraints = [
            ExactCount(name="one_kitchen", room_type_idx=2, target=1),
            RequireAdj(name="kitchen_near_living", type_a_idx=2, type_b_idx=0),
        ]

        graphs = [
            _make_graph([2, 0], [(0, 1, 1)]),   # both satisfied
            _make_graph([0, 0], [(0, 1, 1)]),    # kitchen violated (v=1), adj satisfied
            _make_graph([2, 0], []),              # kitchen satisfied, adj violated (v=1)
            _make_graph([2, 2, 0], [(1, 2, 1)]), # kitchen violated (v=1), adj satisfied
        ]

        calibration = calibrate_from_samples(graphs, constraints)

        assert "one_kitchen" in calibration
        assert "kitchen_near_living" in calibration
        # Both have non-zero violations, so P90 != 1.0
        assert calibration["one_kitchen"] > 0
        assert calibration["kitchen_near_living"] > 0

    def test_empty_graph_list(self):
        """Empty graph list → all P90 = 1.0 (no violations seen)."""
        constraint = ExactCount(name="one_kitchen", room_type_idx=2, target=1)
        calibration = calibrate_from_samples([], [constraint])
        assert calibration["one_kitchen"] == 1.0

    def test_empty_constraint_list(self):
        """No constraints → empty calibration dict."""
        graphs = [_make_graph([2, 0])]
        calibration = calibrate_from_samples(graphs, [])
        assert calibration == {}
