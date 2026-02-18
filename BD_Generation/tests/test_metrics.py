"""Tests for bd_gen.eval.metrics — evaluation metrics."""

from __future__ import annotations

import pytest
import torch

import math

from bd_gen.eval.metrics import (
    _archetype_hash,
    _canonicalize_edge,
    _check_spatial_consistency,
    _clustering_histogram,
    _compute_mmd,
    _conditional_edge_histogram,
    _degree_histogram,
    _graph_dict_to_nx,
    _has_cycle,
    _js_divergence,
    _per_type_degree_histograms,
    _spectral_features,
    _total_variation,
    _wasserstein1_1d_discrete,
    conditional_edge_kl,
    distribution_match,
    diversity,
    graph_structure_mmd,
    mode_coverage,
    novelty,
    per_class_accuracy,
    spatial_transitivity,
    type_conditioned_degree_kl,
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

    def test_returns_js_tv_w1_keys(self):
        g = {
            "num_rooms": 3,
            "node_types": [0, 1, 2],
            "edge_triples": [(0, 1, 2), (1, 2, 3)],
        }
        result = distribution_match([g] * 10, [g] * 10)
        for key in ("node_js", "edge_js", "node_tv", "edge_tv", "rooms_w1"):
            assert key in result

    def test_identical_js_tv_w1_near_zero(self):
        g = {
            "num_rooms": 3,
            "node_types": [0, 1, 2],
            "edge_triples": [(0, 1, 2), (1, 2, 3)],
        }
        result = distribution_match([g] * 10, [g] * 10)
        assert result["node_js"] < 1e-6
        assert result["edge_js"] < 1e-6
        assert result["node_tv"] < 1e-6
        assert result["edge_tv"] < 1e-6
        assert result["rooms_w1"] < 1e-6

    def test_different_distributions_js_tv_positive(self):
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
        assert result["node_js"] > 0.01
        assert result["edge_js"] > 0.01
        assert result["node_tv"] > 0.01
        assert result["edge_tv"] > 0.01
        assert result["rooms_w1"] > 0.01


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


# ---------------------------------------------------------------------------
# Distance primitives: _total_variation, _js_divergence, _wasserstein1_1d_discrete
# ---------------------------------------------------------------------------


class TestTotalVariation:
    def test_identical(self):
        p = [0.25, 0.25, 0.25, 0.25]
        assert _total_variation(p, p) == pytest.approx(0.0)

    def test_disjoint(self):
        assert _total_variation([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)

    def test_symmetric(self):
        p = [0.3, 0.7]
        q = [0.5, 0.5]
        assert _total_variation(p, q) == pytest.approx(_total_variation(q, p))

    def test_bounded_0_1(self):
        p = [0.1, 0.9]
        q = [0.6, 0.4]
        tv = _total_variation(p, q)
        assert 0.0 <= tv <= 1.0

    def test_different_lengths(self):
        p = [0.5, 0.5]
        q = [0.5, 0.3, 0.2]
        tv = _total_variation(p, q)
        assert tv > 0.0


class TestJSDivergence:
    def test_identical(self):
        p = [0.25, 0.25, 0.25, 0.25]
        assert _js_divergence(p, p) == pytest.approx(0.0)

    def test_disjoint(self):
        js = _js_divergence([1.0, 0.0], [0.0, 1.0])
        assert js == pytest.approx(math.log(2), abs=1e-10)

    def test_symmetric(self):
        p = [0.3, 0.7]
        q = [0.5, 0.5]
        assert _js_divergence(p, q) == pytest.approx(_js_divergence(q, p))

    def test_bounded(self):
        p = [0.1, 0.9]
        q = [0.6, 0.4]
        js = _js_divergence(p, q)
        assert 0.0 <= js <= math.log(2)

    def test_different_lengths(self):
        p = [0.5, 0.5]
        q = [0.5, 0.3, 0.2]
        js = _js_divergence(p, q)
        assert js > 0.0

    def test_zero_entries(self):
        """p_k=0 should not cause errors (0*log(0/x)=0 convention)."""
        p = [1.0, 0.0, 0.0]
        q = [0.0, 0.5, 0.5]
        js = _js_divergence(p, q)
        assert js == pytest.approx(math.log(2), abs=1e-10)


class TestWasserstein1:
    def test_identical(self):
        p = [0.25, 0.25, 0.25, 0.25]
        assert _wasserstein1_1d_discrete(p, p) == pytest.approx(0.0)

    def test_shift_by_one(self):
        """Shifting mass one bin costs 1.0."""
        p = [1.0, 0.0, 0.0]
        q = [0.0, 1.0, 0.0]
        assert _wasserstein1_1d_discrete(p, q) == pytest.approx(1.0)

    def test_shift_by_two(self):
        """Shifting mass two bins costs 2.0."""
        p = [1.0, 0.0, 0.0]
        q = [0.0, 0.0, 1.0]
        assert _wasserstein1_1d_discrete(p, q) == pytest.approx(2.0)

    def test_symmetric(self):
        p = [0.3, 0.7]
        q = [0.5, 0.5]
        assert _wasserstein1_1d_discrete(p, q) == pytest.approx(
            _wasserstein1_1d_discrete(q, p)
        )

    def test_different_lengths(self):
        p = [0.5, 0.5]
        q = [0.5, 0.3, 0.2]
        w1 = _wasserstein1_1d_discrete(p, q)
        assert w1 > 0.0


# ---------------------------------------------------------------------------
# _canonicalize_edge
# ---------------------------------------------------------------------------


class TestCanonicalizeEdge:
    def test_no_swap_when_type_i_less(self):
        pair, rel = _canonicalize_edge(0, 3, 2)  # Liv < Bath
        assert pair == (0, 3)
        assert rel == 2

    def test_swap_when_type_i_greater(self):
        # Bath(3) > Liv(0): swap and invert 9-7=2
        pair, rel = _canonicalize_edge(3, 0, 7)
        assert pair == (0, 3)
        assert rel == 2

    def test_equal_types_no_swap(self):
        pair, rel = _canonicalize_edge(3, 3, 4)
        assert pair == (3, 3)
        assert rel == 4

    def test_inverse_symmetry(self):
        pair1, rel1 = _canonicalize_edge(0, 3, 2)
        pair2, rel2 = _canonicalize_edge(3, 0, 7)  # 9-2=7
        assert pair1 == pair2
        assert rel1 == rel2

    def test_all_inverse_pairs(self):
        """Every rel and its inverse (9-rel) canonicalize identically."""
        for rel in range(10):
            pair_fwd, rel_fwd = _canonicalize_edge(0, 5, rel)
            pair_inv, rel_inv = _canonicalize_edge(5, 0, 9 - rel)
            assert pair_fwd == pair_inv
            assert rel_fwd == rel_inv


# ---------------------------------------------------------------------------
# conditional_edge_kl
# ---------------------------------------------------------------------------


def _g(node_types, edge_triples):
    """Minimal graph dict helper."""
    return {
        "num_rooms": len(node_types),
        "node_types": node_types,
        "edge_triples": edge_triples,
    }


class TestConditionalEdgeKL:
    def test_identical_distributions_near_zero(self):
        g = _g([0, 3], [(0, 1, 2)])
        graphs = [g] * 20
        result = conditional_edge_kl(graphs, graphs, min_pair_count=1)
        assert result["conditional_edge_kl_mean"] < 1e-6
        assert result["conditional_edge_kl_weighted"] < 1e-6

    def test_returns_expected_keys(self):
        g = _g([0, 1], [(0, 1, 3)])
        result = conditional_edge_kl([g] * 10, [g] * 10, min_pair_count=1)
        expected = {
            "conditional_edge_kl_mean",
            "conditional_edge_kl_weighted",
            "conditional_edge_js_mean",
            "conditional_edge_js_weighted",
            "conditional_edge_tv_mean",
            "conditional_edge_tv_weighted",
            "num_pairs_evaluated",
        }
        assert expected.issubset(set(result.keys()))

    def test_empty_samples_returns_zeros(self):
        g = _g([0, 3], [(0, 1, 2)])
        result = conditional_edge_kl([], [g] * 10)
        assert result["conditional_edge_kl_mean"] == 0.0
        assert result["conditional_edge_kl_weighted"] == 0.0
        assert result["num_pairs_evaluated"] == 0.0

    def test_empty_training_returns_zeros(self):
        g = _g([0, 3], [(0, 1, 2)])
        result = conditional_edge_kl([g] * 10, [])
        assert result["conditional_edge_kl_mean"] == 0.0

    def test_different_distributions_positive_kl(self):
        # Training: Liv-Bath always left-of (2); Samples: always above (3)
        train = [_g([0, 3], [(0, 1, 2)])] * 20
        samples = [_g([0, 3], [(0, 1, 3)])] * 20
        result = conditional_edge_kl(samples, train, min_pair_count=1)
        assert result["conditional_edge_kl_mean"] > 0.1

    def test_canonicalization_consistency(self):
        # (Liv=0, Bath=3, left-of=2) and (Bath=3, Liv=0, right-of=7) are
        # the same spatial fact; histograms should match.
        g_a = _g([0, 3], [(0, 1, 2)])
        g_b = _g([3, 0], [(0, 1, 7)])
        hist_a, _ = _conditional_edge_histogram([g_a] * 10)
        hist_b, _ = _conditional_edge_histogram([g_b] * 10)
        assert set(hist_a.keys()) == set(hist_b.keys())
        for pair in hist_a:
            for a, b in zip(hist_a[pair], hist_b[pair]):
                assert abs(a - b) < 1e-9

    def test_min_pair_count_filters(self):
        train = [
            _g([0, 3], [(0, 1, 2)]),
            _g([0, 3], [(0, 1, 2)]),
            _g([0, 1], [(0, 1, 3)]),
        ]
        # min=2: only (0,3) qualifies (2 edges); min=1: both qualify
        r_strict = conditional_edge_kl(train, train, min_pair_count=2)
        r_lenient = conditional_edge_kl(train, train, min_pair_count=1)
        assert r_strict["num_pairs_evaluated"] == 1.0
        assert r_lenient["num_pairs_evaluated"] == 2.0

    def test_num_pairs_evaluated(self):
        g1 = _g([0, 3], [(0, 1, 2)])
        g2 = _g([0, 1], [(0, 1, 3)])
        g3 = _g([1, 3], [(0, 1, 4)])
        train = [g1, g2, g3] * 10
        result = conditional_edge_kl(train, train, min_pair_count=1)
        assert result["num_pairs_evaluated"] == 3.0

    def test_same_type_pair(self):
        g = _g([3, 3], [(0, 1, 3)])  # Bath-Bath, above
        result = conditional_edge_kl([g] * 10, [g] * 10, min_pair_count=1)
        assert result["conditional_edge_kl_mean"] < 1e-6

    def test_identical_js_tv_near_zero(self):
        g = _g([0, 3], [(0, 1, 2)])
        result = conditional_edge_kl([g] * 20, [g] * 20, min_pair_count=1)
        assert result["conditional_edge_js_mean"] < 1e-6
        assert result["conditional_edge_tv_mean"] < 1e-6

    def test_different_distributions_js_tv_positive(self):
        train = [_g([0, 3], [(0, 1, 2)])] * 20
        samples = [_g([0, 3], [(0, 1, 3)])] * 20
        result = conditional_edge_kl(samples, train, min_pair_count=1)
        assert result["conditional_edge_js_mean"] > 0.01
        assert result["conditional_edge_tv_mean"] > 0.01


# ---------------------------------------------------------------------------
# conditional_edge_distances_topN
# ---------------------------------------------------------------------------


class TestConditionalEdgeDistancesTopN:
    def test_identical_near_zero(self):
        from bd_gen.eval.metrics import conditional_edge_distances_topN

        g = _g([0, 3], [(0, 1, 2)])
        result = conditional_edge_distances_topN(
            [g] * 20, [g] * 20, top_n=5, min_pair_count=1,
        )
        assert result["conditional_edge_kl_topN_mean"] < 1e-6
        assert result["conditional_edge_js_topN_mean"] < 1e-6
        assert result["conditional_edge_tv_topN_mean"] < 1e-6

    def test_selects_top_n_pairs(self):
        from bd_gen.eval.metrics import conditional_edge_distances_topN

        # 3 distinct pairs with different frequencies
        g1 = _g([0, 3], [(0, 1, 2)])  # pair (0,3)
        g2 = _g([0, 1], [(0, 1, 3)])  # pair (0,1)
        g3 = _g([1, 3], [(0, 1, 4)])  # pair (1,3)
        # (0,3) appears 10x, (0,1) 5x, (1,3) 2x
        train = [g1] * 10 + [g2] * 5 + [g3] * 2
        result = conditional_edge_distances_topN(
            train, train, top_n=2, min_pair_count=1,
        )
        # Should select 2 most frequent pairs
        assert result["num_pairs_evaluated"] == 2.0
        assert result["topN"] == 2.0

    def test_empty_samples(self):
        from bd_gen.eval.metrics import conditional_edge_distances_topN

        g = _g([0, 3], [(0, 1, 2)])
        result = conditional_edge_distances_topN([], [g] * 10)
        assert result["num_pairs_evaluated"] == 0.0

    def test_returns_expected_keys(self):
        from bd_gen.eval.metrics import conditional_edge_distances_topN

        g = _g([0, 3], [(0, 1, 2)])
        result = conditional_edge_distances_topN(
            [g] * 10, [g] * 10, top_n=5, min_pair_count=1,
        )
        expected = {
            "conditional_edge_kl_topN_mean",
            "conditional_edge_kl_topN_weighted",
            "conditional_edge_js_topN_mean",
            "conditional_edge_js_topN_weighted",
            "conditional_edge_tv_topN_mean",
            "conditional_edge_tv_topN_weighted",
            "topN",
            "num_pairs_evaluated",
        }
        assert expected.issubset(set(result.keys()))


# ---------------------------------------------------------------------------
# _has_cycle
# ---------------------------------------------------------------------------


class TestHasCycle:
    def test_no_edges(self):
        assert _has_cycle([[], [], []], 3) is False

    def test_simple_chain(self):
        # 0 -> 1 -> 2 (no cycle)
        assert _has_cycle([[1], [2], []], 3) is False

    def test_simple_cycle(self):
        # 0 -> 1 -> 2 -> 0
        assert _has_cycle([[1], [2], [0]], 3) is True

    def test_self_loop(self):
        assert _has_cycle([[0]], 1) is True

    def test_diamond_no_cycle(self):
        # 0->1, 0->2, 1->3, 2->3 (DAG)
        assert _has_cycle([[1, 2], [3], [3], []], 4) is False


# ---------------------------------------------------------------------------
# _check_spatial_consistency
# ---------------------------------------------------------------------------


class TestCheckSpatialConsistency:
    def test_consistent_left_chain(self):
        # A left-of B (2), B left-of C (2) — consistent (A.x < B.x < C.x)
        g = _g([0, 1, 2], [(0, 1, 2), (1, 2, 2)])
        result = _check_spatial_consistency(g)
        assert result["h_consistent"] is True
        assert result["v_consistent"] is True
        assert result["overall"] is True

    def test_contradictory_horizontal_cycle(self):
        # A left-of B (2), B left-of C (2), C left-of A (2)
        # Creates cycle: A.x < B.x < C.x < A.x — impossible
        g = _g([0, 1, 2], [(0, 1, 2), (1, 2, 2), (0, 2, 7)])  # 7 = right-of
        # 0 left-of 1 => 0.x < 1.x; 1 left-of 2 => 1.x < 2.x
        # 0 right-of 2 => 0.x > 2.x — contradiction with 0.x < 1.x < 2.x
        result = _check_spatial_consistency(g)
        assert result["h_consistent"] is False
        assert result["overall"] is False

    def test_contradictory_vertical_cycle(self):
        # A above B (3), B above C (3), C above A (3)
        # 3=above: A.y > B.y; for (1,2,3): 1 above 2, 1.y > 2.y
        # We need C above A: edge (0,2, 6) means 0 below 2, i.e., 0.y < 2.y
        g = _g([0, 1, 2], [(0, 1, 3), (1, 2, 3), (0, 2, 6)])
        # 0 above 1 => 0.y > 1.y; 1 above 2 => 1.y > 2.y
        # 0 below 2 => 0.y < 2.y — contradiction with 0.y > 1.y > 2.y
        result = _check_spatial_consistency(g)
        assert result["v_consistent"] is False
        assert result["overall"] is False

    def test_inside_surrounding_no_constraints(self):
        # Only inside/surrounding edges — no ordering constraints
        g = _g([0, 1, 2], [(0, 1, 4), (1, 2, 5)])  # 4=inside, 5=surrounding
        result = _check_spatial_consistency(g)
        assert result["h_consistent"] is True
        assert result["v_consistent"] is True
        assert result["overall"] is True

    def test_single_room(self):
        g = _g([0], [])
        result = _check_spatial_consistency(g)
        assert result["overall"] is True

    def test_mixed_axes(self):
        # H-consistent but V-contradictory
        # 0 left-above 1 (rel 0): 0.x < 1.x, 0.y > 1.y
        # 1 above 2 (rel 3): 1.y > 2.y
        # 0 below 2 (rel 6): 0.y < 2.y
        # V: 0.y > 1.y > 2.y, but 0.y < 2.y — contradiction
        # H: 0.x < 1.x — no cycle, just one constraint
        g = _g([0, 1, 2], [(0, 1, 0), (1, 2, 3), (0, 2, 6)])
        result = _check_spatial_consistency(g)
        assert result["h_consistent"] is True
        assert result["v_consistent"] is False
        assert result["overall"] is False


# ---------------------------------------------------------------------------
# spatial_transitivity
# ---------------------------------------------------------------------------


class TestSpatialTransitivity:
    def test_all_consistent(self):
        g = _g([0, 1, 2], [(0, 1, 2), (1, 2, 2)])  # left-of chain
        result = spatial_transitivity([g] * 10)
        assert result["transitivity_score"] == 1.0

    def test_all_contradictory(self):
        g = _g([0, 1, 2], [(0, 1, 2), (1, 2, 2), (0, 2, 7)])
        result = spatial_transitivity([g] * 10)
        assert result["transitivity_score"] == 0.0

    def test_mixed(self):
        good = _g([0, 1, 2], [(0, 1, 2), (1, 2, 2)])
        bad = _g([0, 1, 2], [(0, 1, 2), (1, 2, 2), (0, 2, 7)])
        result = spatial_transitivity([good, good, bad, bad])
        assert abs(result["transitivity_score"] - 0.5) < 1e-9

    def test_empty_input(self):
        result = spatial_transitivity([])
        assert result["transitivity_score"] == 1.0
        assert result["h_consistent"] == 1.0
        assert result["v_consistent"] == 1.0

    def test_returns_three_keys(self):
        g = _g([0, 1], [(0, 1, 2)])
        result = spatial_transitivity([g])
        assert set(result.keys()) == {
            "transitivity_score", "h_consistent", "v_consistent"
        }


# ---------------------------------------------------------------------------
# _per_type_degree_histograms
# ---------------------------------------------------------------------------


class TestPerTypeDegreeHistograms:
    def test_basic(self):
        # 3-node path: 0-1-2, types [0, 1, 0]
        # degrees: node 0 -> 1, node 1 -> 2, node 2 -> 1
        # type 0 has degrees [1, 1], type 1 has degrees [2]
        g = _g([0, 1, 0], [(0, 1, 2), (1, 2, 3)])
        hists, counts = _per_type_degree_histograms([g], n_max=4)
        assert counts[0] == 2  # two nodes of type 0
        assert counts[1] == 1  # one node of type 1
        assert hists[0][1] == pytest.approx(1.0)  # both type-0 nodes have degree 1
        assert hists[1][2] == pytest.approx(1.0)  # type-1 node has degree 2

    def test_empty(self):
        hists, counts = _per_type_degree_histograms([], n_max=8)
        assert hists == {}
        assert counts == {}


# ---------------------------------------------------------------------------
# type_conditioned_degree_kl
# ---------------------------------------------------------------------------


class TestTypeConditionedDegreeKL:
    def test_identical_distributions(self):
        g = _g([0, 1, 2], [(0, 1, 2), (1, 2, 3)])
        graphs = [g] * 30
        result = type_conditioned_degree_kl(graphs, graphs, n_max=4, min_type_count=1)
        assert result["degree_kl_per_type_mean"] < 1e-6
        assert result["degree_kl_per_type_weighted"] < 1e-6

    def test_different_distributions(self):
        # Training: type 0 nodes have degree 1
        train = [_g([0, 1], [(0, 1, 2)])] * 30
        # Samples: type 0 nodes have degree 3 (connected to all others)
        samples = [_g([0, 1, 2, 3], [(0, 1, 2), (0, 2, 3), (0, 3, 7)])] * 30
        result = type_conditioned_degree_kl(samples, train, n_max=8, min_type_count=1)
        assert result["degree_kl_per_type_mean"] > 0.1

    def test_min_type_count_filters(self):
        g = _g([0, 1], [(0, 1, 2)])
        # Only 5 graphs, so 5 nodes of each type
        result_strict = type_conditioned_degree_kl(
            [g] * 5, [g] * 5, n_max=4, min_type_count=10,
        )
        result_lenient = type_conditioned_degree_kl(
            [g] * 5, [g] * 5, n_max=4, min_type_count=1,
        )
        assert result_strict["num_types_evaluated"] == 0.0  # all filtered
        assert result_lenient["num_types_evaluated"] == 2.0  # both types pass

    def test_empty_samples(self):
        g = _g([0, 1], [(0, 1, 2)])
        result = type_conditioned_degree_kl([], [g] * 10, n_max=4)
        assert result["degree_kl_per_type_mean"] == 0.0

    def test_empty_reference(self):
        g = _g([0, 1], [(0, 1, 2)])
        result = type_conditioned_degree_kl([g] * 10, [], n_max=4)
        assert result["degree_kl_per_type_mean"] == 0.0

    def test_returns_expected_keys(self):
        g = _g([0, 1], [(0, 1, 2)])
        result = type_conditioned_degree_kl([g] * 10, [g] * 10, n_max=4, min_type_count=1)
        expected = {
            "degree_kl_per_type_mean",
            "degree_kl_per_type_weighted",
            "degree_js_per_type_mean",
            "degree_js_per_type_weighted",
            "degree_tv_per_type_mean",
            "degree_tv_per_type_weighted",
            "num_types_evaluated",
        }
        assert expected.issubset(set(result.keys()))

    def test_identical_js_tv_near_zero(self):
        g = _g([0, 1, 2], [(0, 1, 2), (1, 2, 3)])
        graphs = [g] * 30
        result = type_conditioned_degree_kl(graphs, graphs, n_max=4, min_type_count=1)
        assert result["degree_js_per_type_mean"] < 1e-6
        assert result["degree_tv_per_type_mean"] < 1e-6

    def test_different_distributions_js_tv_positive(self):
        train = [_g([0, 1], [(0, 1, 2)])] * 30
        samples = [_g([0, 1, 2, 3], [(0, 1, 2), (0, 2, 3), (0, 3, 7)])] * 30
        result = type_conditioned_degree_kl(samples, train, n_max=8, min_type_count=1)
        assert result["degree_js_per_type_mean"] > 0.001
        assert result["degree_tv_per_type_mean"] > 0.001


# ---------------------------------------------------------------------------
# _archetype_hash
# ---------------------------------------------------------------------------


class TestArchetypeHash:
    def test_sorted_output(self):
        g = _g([3, 0, 1], [])
        assert _archetype_hash(g) == (0, 1, 3)

    def test_same_types_different_order(self):
        g1 = _g([0, 1, 2], [])
        g2 = _g([2, 0, 1], [])
        assert _archetype_hash(g1) == _archetype_hash(g2)

    def test_duplicate_types(self):
        g = _g([3, 3, 0], [])
        assert _archetype_hash(g) == (0, 3, 3)


# ---------------------------------------------------------------------------
# mode_coverage
# ---------------------------------------------------------------------------


class TestModeCoverage:
    def test_perfect_coverage(self):
        training = [_g([0, 1], []), _g([0, 2], []), _g([1, 2], [])]
        samples = [_g([0, 1], []), _g([0, 2], []), _g([1, 2], [])]
        result = mode_coverage(samples, training)
        assert result["mode_coverage"] == 1.0
        assert result["mode_coverage_weighted"] == 1.0

    def test_partial_coverage(self):
        training = [
            _g([0, 1], []),
            _g([0, 2], []),
            _g([1, 2], []),
            _g([0, 1, 2], []),
        ]
        # Only one archetype covered
        samples = [_g([0, 1], [])] * 10
        result = mode_coverage(samples, training)
        assert abs(result["mode_coverage"] - 0.25) < 1e-9  # 1/4

    def test_no_overlap(self):
        training = [_g([0, 1], [])] * 10
        samples = [_g([2, 3], [])] * 10
        result = mode_coverage(samples, training)
        assert result["mode_coverage"] == 0.0
        assert result["mode_coverage_weighted"] == 0.0

    def test_weighted_vs_unweighted(self):
        # Training: archetype (0,1) appears 9 times, (0,2) appears 1 time
        training = [_g([0, 1], [])] * 9 + [_g([0, 2], [])]
        # Samples cover only (0,1)
        samples = [_g([0, 1], [])] * 5
        result = mode_coverage(samples, training)
        assert abs(result["mode_coverage"] - 0.5) < 1e-9  # 1/2 archetypes
        assert abs(result["mode_coverage_weighted"] - 0.9) < 1e-9  # 9/10 mass

    def test_empty_samples(self):
        training = [_g([0, 1], [])]
        result = mode_coverage([], training)
        assert result["mode_coverage"] == 0.0

    def test_empty_training(self):
        samples = [_g([0, 1], [])]
        result = mode_coverage(samples, [])
        assert result["mode_coverage"] == 0.0

    def test_num_modes(self):
        training = [_g([0, 1], []), _g([0, 2], []), _g([1, 2], [])]
        samples = [_g([0, 1], []), _g([3, 4], [])]  # 1 overlap + 1 novel
        result = mode_coverage(samples, training)
        assert result["num_training_modes"] == 3.0
        assert result["num_sample_modes"] == 2.0

    def test_returns_four_keys(self):
        g = _g([0, 1], [])
        result = mode_coverage([g], [g])
        assert set(result.keys()) == {
            "mode_coverage",
            "mode_coverage_weighted",
            "num_training_modes",
            "num_sample_modes",
        }


# ---------------------------------------------------------------------------
# Stratified metrics: validity_by_num_rooms, spatial_transitivity_by_num_rooms,
# edge_present_rate_by_num_rooms
# ---------------------------------------------------------------------------


class TestValidityByNumRooms:
    def test_groups_by_num_rooms(self):
        from bd_gen.eval.metrics import validity_by_num_rooms

        g3 = _g([0, 1, 2], [(0, 1, 2), (1, 2, 3)])
        g4 = _g([0, 1, 2, 3], [(0, 1, 2), (1, 2, 3), (2, 3, 0)])
        validity_results = [
            {"overall": True, "connected": True, "valid_types": True, "no_mask_tokens": True},
            {"overall": True, "connected": True, "valid_types": True, "no_mask_tokens": True},
            {"overall": False, "connected": False, "valid_types": True, "no_mask_tokens": True},
        ]
        graph_dicts = [g3, g3, g4]
        result = validity_by_num_rooms(validity_results, graph_dicts)
        assert "3" in result
        assert "4" in result
        assert result["3"]["overall"] == 1.0
        assert result["4"]["overall"] == 0.0

    def test_empty_input(self):
        from bd_gen.eval.metrics import validity_by_num_rooms

        assert validity_by_num_rooms([], []) == {}


class TestSpatialTransitivityByNumRooms:
    def test_groups_by_num_rooms(self):
        from bd_gen.eval.metrics import spatial_transitivity_by_num_rooms

        good3 = _g([0, 1, 2], [(0, 1, 2), (1, 2, 2)])
        bad4 = _g([0, 1, 2, 3], [(0, 1, 2), (1, 2, 2), (0, 2, 7)])
        result = spatial_transitivity_by_num_rooms([good3, good3, bad4])
        assert "3" in result
        assert result["3"]["transitivity_score"] == 1.0
        assert "4" in result
        assert result["4"]["transitivity_score"] == 0.0

    def test_empty_input(self):
        from bd_gen.eval.metrics import spatial_transitivity_by_num_rooms

        assert spatial_transitivity_by_num_rooms([]) == {}


class TestEdgePresentRateByNumRooms:
    def test_known_rates(self):
        from bd_gen.eval.metrics import edge_present_rate_by_num_rooms

        # 3 rooms: E_possible = 3, 2 edges present -> rate = 2/3
        g3 = _g([0, 1, 2], [(0, 1, 2), (1, 2, 3)])
        # 4 rooms: E_possible = 6, 3 edges present -> rate = 0.5
        g4 = _g([0, 1, 2, 3], [(0, 1, 2), (1, 2, 3), (2, 3, 0)])
        result = edge_present_rate_by_num_rooms([g3, g4])
        assert result["3"] == pytest.approx(2 / 3)
        assert result["4"] == pytest.approx(0.5)

    def test_averages_within_group(self):
        from bd_gen.eval.metrics import edge_present_rate_by_num_rooms

        # Two 3-room graphs with different edge counts
        g3a = _g([0, 1, 2], [(0, 1, 2)])  # 1/3
        g3b = _g([0, 1, 2], [(0, 1, 2), (0, 2, 3), (1, 2, 0)])  # 3/3
        result = edge_present_rate_by_num_rooms([g3a, g3b])
        assert result["3"] == pytest.approx((1 / 3 + 1.0) / 2)

    def test_empty_input(self):
        from bd_gen.eval.metrics import edge_present_rate_by_num_rooms

        assert edge_present_rate_by_num_rooms([]) == {}


# ---------------------------------------------------------------------------
# _aggregate_multi_seed (from evaluate.py)
# ---------------------------------------------------------------------------


class TestAggregateMultiSeed:
    def test_mean_std_basic(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from evaluate import _aggregate_multi_seed

        per_seed = {
            42: {"eval/validity_rate": 0.9, "eval/diversity": 0.8},
            123: {"eval/validity_rate": 0.95, "eval/diversity": 0.85},
        }
        summary = _aggregate_multi_seed(per_seed)
        assert "eval/validity_rate" in summary
        assert summary["eval/validity_rate"]["mean"] == pytest.approx(0.925)
        assert summary["eval/validity_rate"]["std"] > 0.0

    def test_nested_dicts_flattened(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from evaluate import _aggregate_multi_seed

        per_seed = {
            42: {"eval/x": 1.0, "stratified": {"3": {"val": 0.5}}},
            123: {"eval/x": 2.0, "stratified": {"3": {"val": 0.7}}},
        }
        summary = _aggregate_multi_seed(per_seed)
        assert "stratified/3/val" in summary
        assert summary["stratified/3/val"]["mean"] == pytest.approx(0.6)

    def test_single_seed_zero_std(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from evaluate import _aggregate_multi_seed

        per_seed = {42: {"eval/a": 0.5}}
        summary = _aggregate_multi_seed(per_seed)
        assert summary["eval/a"]["mean"] == pytest.approx(0.5)
        assert summary["eval/a"]["std"] == pytest.approx(0.0)
