"""Tests for bd_gen.eval.metrics — evaluation metrics."""

from __future__ import annotations

import torch

from bd_gen.eval.metrics import (
    distribution_match,
    diversity,
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
