"""Tests for bd_gen.eval.metrics — evaluation metrics."""

from __future__ import annotations

import pytest
import torch

from bd_gen.eval.metrics import (
    _clustering_histogram,
    _compute_mmd,
    _degree_histogram,
    _graph_dict_to_nx,
    _spectral_features,
    distribution_match,
    diversity,
    graph_structure_mmd,
    novelty,
    per_class_accuracy,
    validity_rate,
)

# ---------------------------------------------------------------------------
# validity_rate
# ---------------------------------------------------------------------------


class TestValidityRate:
    def test_all_valid(self):
        results = [{"overall": True}] * 5
        assert validity_rate(results) == 1.0

    def test_none_valid(self):
        results = [{"overall": False}] * 5
        assert validity_rate(results) == 0.0

    def test_mixed(self):
        results = [{"overall": True}] * 3 + [{"overall": False}] * 2
        assert abs(validity_rate(results) - 0.6) < 1e-9

    def test_empty_list(self):
        assert validity_rate([]) == 0.0


# ---------------------------------------------------------------------------
# diversity
# ---------------------------------------------------------------------------


class TestDiversity:
    def test_all_unique(self):
        graphs = [
            {"num_rooms": 2, "node_types": [0, 1], "edge_triples": [(0, 1, i)]}
            for i in range(5)
        ]
        assert diversity(graphs) == 1.0

    def test_all_identical(self):
        graph = {"num_rooms": 2, "node_types": [0, 1], "edge_triples": [(0, 1, 2)]}
        graphs = [graph] * 5
        assert abs(diversity(graphs) - 0.2) < 1e-9

    def test_some_duplicates(self):
        g1 = {"num_rooms": 2, "node_types": [0, 1], "edge_triples": [(0, 1, 2)]}
        g2 = {"num_rooms": 2, "node_types": [0, 1], "edge_triples": [(0, 1, 3)]}
        g3 = {"num_rooms": 3, "node_types": [0, 1, 2], "edge_triples": []}
        graphs = [g1, g1, g2, g2, g3]
        assert abs(diversity(graphs) - 0.6) < 1e-9

    def test_single_sample(self):
        graph = {"num_rooms": 1, "node_types": [0], "edge_triples": []}
        assert diversity([graph]) == 1.0

    def test_empty(self):
        assert diversity([]) == 0.0


# ---------------------------------------------------------------------------
# novelty
# ---------------------------------------------------------------------------


class TestNovelty:
    def test_all_novel(self):
        samples = [
            {"num_rooms": 2, "node_types": [0, 1], "edge_triples": [(0, 1, i)]}
            for i in range(4)
        ]
        training = [
            {"num_rooms": 2, "node_types": [0, 1], "edge_triples": [(0, 1, i)]}
            for i in range(4, 8)
        ]
        assert novelty(samples, training) == 1.0

    def test_none_novel(self):
        graph = {"num_rooms": 2, "node_types": [0, 1], "edge_triples": [(0, 1, 2)]}
        samples = [graph] * 4
        training = [graph] * 10
        assert novelty(samples, training) == 0.0

    def test_mixed_novelty(self):
        shared = {"num_rooms": 2, "node_types": [0, 1], "edge_triples": [(0, 1, 2)]}
        unique1 = {"num_rooms": 2, "node_types": [0, 1], "edge_triples": [(0, 1, 5)]}
        unique2 = {"num_rooms": 3, "node_types": [0, 1, 2], "edge_triples": []}

        samples = [shared, shared, unique1, unique2]
        training = [shared]

        assert abs(novelty(samples, training) - 0.5) < 1e-9

    def test_empty_samples(self):
        training = [{"num_rooms": 1, "node_types": [0], "edge_triples": []}]
        assert novelty([], training) == 0.0

    def test_empty_training(self):
        samples = [{"num_rooms": 1, "node_types": [0], "edge_triples": []}]
        assert novelty(samples, []) == 1.0


# ---------------------------------------------------------------------------
# distribution_match
# ---------------------------------------------------------------------------


class TestDistributionMatch:
    def test_identical_distributions(self):
        g = {
            "num_rooms": 3,
            "node_types": [0, 1, 2],
            "edge_triples": [(0, 1, 2), (1, 2, 3)],
        }
        graphs = [g] * 10

        result = distribution_match(graphs, graphs)
        assert result["node_kl"] < 1e-6
        assert result["edge_kl"] < 1e-6
        assert result["num_rooms_kl"] < 1e-6

    def test_different_distributions(self):
        samples = [
            {"num_rooms": 2, "node_types": [0, 0], "edge_triples": [(0, 1, 0)]}
        ] * 10
        t = {
            "num_rooms": 4,
            "node_types": [5, 6, 7, 8],
            "edge_triples": [(0, 1, 5), (2, 3, 9)],
        }
        training = [t] * 10

        result = distribution_match(samples, training)
        assert result["node_kl"] > 0.1
        assert result["edge_kl"] > 0.1
        assert result["num_rooms_kl"] > 0.1

    def test_empty_samples(self):
        training = [{"num_rooms": 2, "node_types": [0, 1], "edge_triples": []}]
        result = distribution_match([], training)
        assert result["node_kl"] == 0.0
        assert result["edge_kl"] == 0.0
        assert result["num_rooms_kl"] == 0.0

    def test_empty_training(self):
        samples = [{"num_rooms": 2, "node_types": [0, 1], "edge_triples": []}]
        result = distribution_match(samples, [])
        assert result["node_kl"] == 0.0


# ---------------------------------------------------------------------------
# per_class_accuracy
# ---------------------------------------------------------------------------


class TestPerClassAccuracy:
    def test_perfect_predictions(self):
        preds = torch.tensor([[0, 1, 2], [3, 4, 5]])
        targets = torch.tensor([[0, 1, 2], [3, 4, 5]])
        mask = torch.ones(2, 3, dtype=torch.bool)

        result = per_class_accuracy(preds, targets, mask)
        assert result["overall"] == 1.0
        for cls_acc in result["per_class"].values():
            assert cls_acc == 1.0

    def test_zero_accuracy(self):
        preds = torch.tensor([[0, 0, 0]])
        targets = torch.tensor([[1, 2, 3]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        result = per_class_accuracy(preds, targets, mask)
        assert result["overall"] == 0.0

    def test_masked_positions_excluded(self):
        preds = torch.tensor([[0, 99, 2]])   # position 1 is wrong
        targets = torch.tensor([[0, 1, 2]])
        mask = torch.tensor([[True, False, True]])  # Mask out position 1

        result = per_class_accuracy(preds, targets, mask)
        assert result["overall"] == 1.0  # Only positions 0 and 2 count

    def test_per_class_breakdown(self):
        # class 0: 1/2 correct, class 1: 1/2 correct
        preds = torch.tensor([[0, 1, 1, 0]])
        targets = torch.tensor([[0, 1, 0, 1]])
        mask = torch.ones(1, 4, dtype=torch.bool)

        result = per_class_accuracy(preds, targets, mask)
        assert result["overall"] == 0.5
        assert result["per_class"][0] == 0.5
        assert result["per_class"][1] == 0.5

    def test_empty_mask(self):
        preds = torch.tensor([[0, 1, 2]])
        targets = torch.tensor([[0, 1, 2]])
        mask = torch.zeros(1, 3, dtype=torch.bool)

        result = per_class_accuracy(preds, targets, mask)
        assert result["overall"] == 0.0
        assert result["per_class"] == {}


# ---------------------------------------------------------------------------
# _graph_dict_to_nx
# ---------------------------------------------------------------------------


def _make_triangle():
    """3-node triangle graph dict."""
    return {
        "num_rooms": 3,
        "node_types": [0, 1, 2],
        "edge_triples": [(0, 1, 2), (0, 2, 3), (1, 2, 0)],
    }


def _make_path3():
    """3-node path: 0-1-2."""
    return {
        "num_rooms": 3,
        "node_types": [0, 1, 2],
        "edge_triples": [(0, 1, 2), (1, 2, 3)],
    }


def _make_k4():
    """Complete graph K4."""
    return {
        "num_rooms": 4,
        "node_types": [0, 1, 2, 3],
        "edge_triples": [
            (0, 1, 0), (0, 2, 1), (0, 3, 2),
            (1, 2, 3), (1, 3, 4), (2, 3, 5),
        ],
    }


def _make_star3():
    """Star graph: node 0 connected to 1, 2, 3."""
    return {
        "num_rooms": 4,
        "node_types": [0, 1, 2, 3],
        "edge_triples": [(0, 1, 0), (0, 2, 1), (0, 3, 2)],
    }


class TestGraphDictToNx:
    def test_triangle_conversion(self):
        g = _graph_dict_to_nx(_make_triangle())
        assert g.number_of_nodes() == 3
        assert g.number_of_edges() == 3

    def test_empty_graph(self):
        g = _graph_dict_to_nx({"num_rooms": 0, "node_types": [], "edge_triples": []})
        assert g.number_of_nodes() == 0
        assert g.number_of_edges() == 0

    def test_single_node(self):
        g = _graph_dict_to_nx({"num_rooms": 1, "node_types": [0], "edge_triples": []})
        assert g.number_of_nodes() == 1
        assert g.number_of_edges() == 0

    def test_edge_types_ignored(self):
        """Different edge types produce same topology."""
        g1 = _graph_dict_to_nx(
            {"num_rooms": 2, "node_types": [0, 1], "edge_triples": [(0, 1, 0)]}
        )
        g2 = _graph_dict_to_nx(
            {"num_rooms": 2, "node_types": [0, 1], "edge_triples": [(0, 1, 9)]}
        )
        assert set(g1.edges()) == set(g2.edges())


# ---------------------------------------------------------------------------
# _degree_histogram
# ---------------------------------------------------------------------------


class TestDegreeHistogram:
    def test_k4_all_degree_3(self):
        g = _graph_dict_to_nx(_make_k4())
        hist = _degree_histogram(g, n_max=8)
        assert len(hist) == 8
        assert hist[3] == pytest.approx(1.0)
        assert sum(hist) == pytest.approx(1.0)

    def test_path3(self):
        g = _graph_dict_to_nx(_make_path3())
        hist = _degree_histogram(g, n_max=8)
        # degrees: 1, 2, 1
        assert hist[1] == pytest.approx(2 / 3)
        assert hist[2] == pytest.approx(1 / 3)

    def test_single_node(self):
        g = _graph_dict_to_nx({"num_rooms": 1, "node_types": [0], "edge_triples": []})
        hist = _degree_histogram(g, n_max=8)
        assert hist[0] == pytest.approx(1.0)

    def test_length_matches_n_max(self):
        g = _graph_dict_to_nx(_make_triangle())
        for n_max in [4, 8, 14]:
            hist = _degree_histogram(g, n_max=n_max)
            assert len(hist) == n_max


# ---------------------------------------------------------------------------
# _clustering_histogram
# ---------------------------------------------------------------------------


class TestClusteringHistogram:
    def test_complete_graph(self):
        """K4: all clustering coefficients = 1.0 -> last bin."""
        g = _graph_dict_to_nx(_make_k4())
        hist = _clustering_histogram(g, n_bins=10)
        assert len(hist) == 10
        assert hist[-1] == pytest.approx(1.0)
        assert sum(hist) == pytest.approx(1.0)

    def test_star_graph(self):
        """Star: center cc=0, leaves cc=0 -> first bin = 1.0."""
        g = _graph_dict_to_nx(_make_star3())
        hist = _clustering_histogram(g, n_bins=10)
        assert hist[0] == pytest.approx(1.0)

    def test_sums_to_one(self):
        g = _graph_dict_to_nx(_make_triangle())
        hist = _clustering_histogram(g, n_bins=10)
        assert sum(hist) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _spectral_features
# ---------------------------------------------------------------------------


class TestSpectralFeatures:
    def test_single_node(self):
        g = _graph_dict_to_nx({"num_rooms": 1, "node_types": [0], "edge_triples": []})
        spec = _spectral_features(g, n_max=8)
        assert len(spec) == 8
        assert all(v == 0.0 for v in spec)

    def test_output_length(self):
        g = _graph_dict_to_nx(_make_k4())
        for n_max in [4, 8, 14]:
            spec = _spectral_features(g, n_max=n_max)
            assert len(spec) == n_max

    def test_complete_graph_spectrum(self):
        """K_n normalized Laplacian: eigenvalue 0 (once) and n/(n-1) (n-1 times)."""
        g = _graph_dict_to_nx(_make_k4())
        spec = _spectral_features(g, n_max=8)
        # First eigenvalue should be ~0
        assert abs(spec[0]) < 1e-6
        # Next 3 eigenvalues should be ~4/3 for K4
        for v in spec[1:4]:
            assert abs(v - 4 / 3) < 1e-6

    def test_disconnected_graph(self):
        """Two components -> at least two zero eigenvalues."""
        gd = {
            "num_rooms": 4,
            "node_types": [0, 1, 2, 3],
            "edge_triples": [(0, 1, 0), (2, 3, 1)],  # two disconnected pairs
        }
        g = _graph_dict_to_nx(gd)
        spec = _spectral_features(g, n_max=8)
        zero_count = sum(1 for v in spec[:4] if abs(v) < 1e-6)
        assert zero_count >= 2


# ---------------------------------------------------------------------------
# _compute_mmd
# ---------------------------------------------------------------------------


class TestComputeMMD:
    def test_identical_distributions(self):
        data = [[1.0, 0.0, 0.0]] * 20
        assert _compute_mmd(data, data) < 1e-6

    def test_different_distributions(self):
        x = [[1.0, 0.0, 0.0]] * 20
        y = [[0.0, 0.0, 1.0]] * 20
        assert _compute_mmd(x, y) > 0.01

    def test_empty_input(self):
        assert _compute_mmd([], [[1.0]]) == 0.0
        assert _compute_mmd([[1.0]], []) == 0.0


# ---------------------------------------------------------------------------
# graph_structure_mmd (integration)
# ---------------------------------------------------------------------------


class TestGraphStructureMMD:
    def test_identical_sets(self):
        graphs = [_make_triangle(), _make_k4(), _make_star3()] * 10
        result = graph_structure_mmd(graphs, graphs, n_max=8)
        assert result["mmd_degree"] < 1e-6
        assert result["mmd_clustering"] < 1e-6
        assert result["mmd_spectral"] < 1e-6

    def test_different_sets(self):
        samples = [_make_k4()] * 20  # all complete graphs
        reference = [_make_path3()] * 20  # all paths
        result = graph_structure_mmd(samples, reference, n_max=8)
        assert result["mmd_degree"] > 0.001
        assert result["mmd_spectral"] > 0.001

    def test_returns_three_keys(self):
        graphs = [_make_triangle()] * 5
        result = graph_structure_mmd(graphs, graphs, n_max=8)
        assert set(result.keys()) == {"mmd_degree", "mmd_clustering", "mmd_spectral"}

    def test_empty_samples(self):
        ref = [_make_triangle()] * 5
        result = graph_structure_mmd([], ref, n_max=8)
        assert result == {"mmd_degree": 0.0, "mmd_clustering": 0.0, "mmd_spectral": 0.0}
