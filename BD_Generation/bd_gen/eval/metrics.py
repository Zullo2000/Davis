"""Evaluation metrics for generated bubble diagrams.

All functions operate on detokenized graph dicts (output of
``bd_gen.data.tokenizer.detokenize``), except ``per_class_accuracy``
which operates on raw tensors for training-time monitoring.

Spec deviation: ``novelty`` uses exact-match (hash-based) instead of
graph edit distance for performance. GED is impractical at scale
(1000 samples x 64K training set).
"""

from __future__ import annotations

import math
from collections import Counter

from torch import Tensor

from bd_gen.data.vocab import EDGE_TYPES, NODE_TYPES

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validity_rate(results: list[dict]) -> float:
    """Fraction of validity results where ``overall`` is True.

    Args:
        results: List of dicts from ``check_validity`` / ``check_validity_batch``.

    Returns:
        Float in [0, 1]. Returns 0.0 for empty input.
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r["overall"]) / len(results)


def diversity(graph_dicts: list[dict]) -> float:
    """Fraction of unique graphs among the samples.

    Two graphs are identical if they have the same num_rooms, the same
    node_types sequence, and the same set of edge triples.

    Args:
        graph_dicts: List of dicts with ``num_rooms``, ``node_types``,
            ``edge_triples`` keys.

    Returns:
        Float in (0, 1]. Returns 0.0 for empty input.
    """
    if not graph_dicts:
        return 0.0
    hashes = {_graph_hash(g) for g in graph_dicts}
    return len(hashes) / len(graph_dicts)


def novelty(
    samples: list[dict],
    training_set: list[dict],
) -> float:
    """Fraction of generated samples not present in the training set.

    Uses exact-match comparison via graph hashing. O(N + M) where N is
    the number of samples and M is the training set size.

    Args:
        samples: Generated graph dicts.
        training_set: Training graph dicts.

    Returns:
        Float in [0, 1]. Returns 0.0 for empty samples.
    """
    if not samples:
        return 0.0

    train_hashes = {_graph_hash(g) for g in training_set}
    novel_count = sum(1 for g in samples if _graph_hash(g) not in train_hashes)
    return novel_count / len(samples)


def distribution_match(
    samples: list[dict],
    training_set: list[dict],
) -> dict[str, float]:
    """KL divergence between sample and training distributions.

    Computes KL(samples || training) for three distributions:
    node types, edge types, and number of rooms.

    Args:
        samples: Generated graph dicts.
        training_set: Training graph dicts.

    Returns:
        Dict with ``node_kl``, ``edge_kl``, ``num_rooms_kl`` (floats >= 0).
        Returns all zeros if either input is empty.
    """
    if not samples or not training_set:
        return {"node_kl": 0.0, "edge_kl": 0.0, "num_rooms_kl": 0.0}

    # Node type histograms
    sample_node_hist = _node_histogram(samples)
    train_node_hist = _node_histogram(training_set)
    node_kl = _kl_divergence(sample_node_hist, train_node_hist)

    # Edge type histograms (spatial relationships only, indices 0-9)
    sample_edge_hist = _edge_histogram(samples)
    train_edge_hist = _edge_histogram(training_set)
    edge_kl = _kl_divergence(sample_edge_hist, train_edge_hist)

    # Num rooms histograms
    sample_rooms_hist = _num_rooms_histogram(samples)
    train_rooms_hist = _num_rooms_histogram(training_set)
    num_rooms_kl = _kl_divergence(sample_rooms_hist, train_rooms_hist)

    return {
        "node_kl": node_kl,
        "edge_kl": edge_kl,
        "num_rooms_kl": num_rooms_kl,
    }


def per_class_accuracy(
    predictions: Tensor,
    targets: Tensor,
    mask: Tensor,
) -> dict:
    """Per-class accuracy at masked positions.

    Args:
        predictions: ``(B, S)`` long tensor of predicted indices.
        targets: ``(B, S)`` long tensor of ground truth indices.
        mask: ``(B, S)`` bool tensor of positions to evaluate.

    Returns:
        Dict with ``overall`` (float) and ``per_class`` (dict mapping
        class index to accuracy float).
    """
    masked_preds = predictions[mask]
    masked_targets = targets[mask]

    if masked_targets.numel() == 0:
        return {"overall": 0.0, "per_class": {}}

    overall = float((masked_preds == masked_targets).float().mean().item())

    per_class: dict[int, float] = {}
    unique_classes = masked_targets.unique().tolist()
    for cls in unique_classes:
        cls_mask = masked_targets == cls
        if cls_mask.any():
            cls_acc = float((masked_preds[cls_mask] == cls).float().mean().item())
            per_class[int(cls)] = cls_acc

    return {"overall": overall, "per_class": per_class}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _graph_hash(graph_dict: dict) -> tuple:
    """Canonical hash for a graph dict.

    Sorts edge triples for order-independence (they should already be
    sorted by (i,j) from detokenize, but sort defensively).
    """
    return (
        graph_dict["num_rooms"],
        tuple(graph_dict["node_types"]),
        tuple(sorted(graph_dict["edge_triples"])),
    )


def _node_histogram(graph_dicts: list[dict]) -> list[float]:
    """Normalized histogram of node types (13 bins)."""
    counts = Counter()
    total = 0
    for g in graph_dicts:
        for nt in g["node_types"]:
            counts[nt] += 1
            total += 1
    if total == 0:
        return [1.0 / len(NODE_TYPES)] * len(NODE_TYPES)
    return [counts.get(i, 0) / total for i in range(len(NODE_TYPES))]


def _edge_histogram(graph_dicts: list[dict]) -> list[float]:
    """Normalized histogram of edge relationship types (10 bins).

    Only counts spatial relationships (indices 0-9), not no-edge.
    Detokenize already filters out no-edge, so all edge_triples have
    rel in [0, 9].
    """
    counts = Counter()
    total = 0
    for g in graph_dicts:
        for _i, _j, rel in g["edge_triples"]:
            counts[rel] += 1
            total += 1
    if total == 0:
        return [1.0 / len(EDGE_TYPES)] * len(EDGE_TYPES)
    return [counts.get(i, 0) / total for i in range(len(EDGE_TYPES))]


def _num_rooms_histogram(graph_dicts: list[dict]) -> list[float]:
    """Normalized histogram of num_rooms values.

    Bins correspond to num_rooms = 1, 2, ..., max_rooms.
    """
    counts = Counter()
    max_rooms = 0
    for g in graph_dicts:
        nr = g["num_rooms"]
        counts[nr] += 1
        if nr > max_rooms:
            max_rooms = nr
    if not counts:
        return [1.0]
    total = sum(counts.values())
    return [counts.get(i, 0) / total for i in range(1, max_rooms + 1)]


def _kl_divergence(p: list[float], q: list[float]) -> float:
    """KL(p || q) with epsilon smoothing to avoid log(0).

    If p and q have different lengths, the shorter one is padded with
    zeros (which contribute 0 to KL since 0 * log(0/q) = 0).
    """
    eps = 1e-10
    max_len = max(len(p), len(q))

    kl = 0.0
    for i in range(max_len):
        pi = p[i] if i < len(p) else 0.0
        qi = q[i] if i < len(q) else 0.0
        if pi > eps:
            kl += pi * math.log(pi / (qi + eps))

    return kl
