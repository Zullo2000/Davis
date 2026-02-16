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

import networkx as nx
import numpy as np
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


def graph_structure_mmd(
    samples: list[dict],
    reference: list[dict],
    *,
    n_max: int = 8,
    sigma: float | None = None,
) -> dict[str, float]:
    """MMD-based structural comparison between generated and reference graphs.

    Computes Maximum Mean Discrepancy with Gaussian RBF kernel for three
    structural statistics, following the evaluation protocol of DiGress,
    MELD, and GraphARM. These metrics measure **topology only** (edge types
    are ignored).

    Args:
        samples: Generated graph dicts.
        reference: Reference (training/test) graph dicts.
        n_max: Maximum number of nodes (for histogram padding). Defaults to 8.
        sigma: Bandwidth for Gaussian RBF kernel. ``None`` = median heuristic.

    Returns:
        Dict with ``mmd_degree``, ``mmd_clustering``, ``mmd_spectral``
        (floats >= 0). Returns all zeros if either input is empty.
    """
    zeros = {"mmd_degree": 0.0, "mmd_clustering": 0.0, "mmd_spectral": 0.0}
    if not samples or not reference:
        return zeros

    # Convert to networkx (skip degenerate graphs)
    sample_graphs = [
        _graph_dict_to_nx(g) for g in samples if g.get("num_rooms", 0) > 0
    ]
    ref_graphs = [
        _graph_dict_to_nx(g) for g in reference if g.get("num_rooms", 0) > 0
    ]
    if not sample_graphs or not ref_graphs:
        return zeros

    # Extract features
    s_deg = [_degree_histogram(g, n_max) for g in sample_graphs]
    r_deg = [_degree_histogram(g, n_max) for g in ref_graphs]
    s_clust = [_clustering_histogram(g) for g in sample_graphs]
    r_clust = [_clustering_histogram(g) for g in ref_graphs]
    s_spec = [_spectral_features(g, n_max) for g in sample_graphs]
    r_spec = [_spectral_features(g, n_max) for g in ref_graphs]

    return {
        "mmd_degree": _compute_mmd(s_deg, r_deg, sigma=sigma),
        "mmd_clustering": _compute_mmd(s_clust, r_clust, sigma=sigma),
        "mmd_spectral": _compute_mmd(s_spec, r_spec, sigma=sigma),
    }


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


# ---------------------------------------------------------------------------
# MMD graph structure helpers
# ---------------------------------------------------------------------------


def _graph_dict_to_nx(graph_dict: dict) -> nx.Graph:
    """Convert graph dict to undirected networkx Graph (topology only).

    Edge types are ignored — only adjacency matters for structural metrics.
    """
    g = nx.Graph()
    num_rooms = graph_dict.get("num_rooms", 0)
    if num_rooms == 0:
        return g
    g.add_nodes_from(range(num_rooms))
    for i, j, _rel in graph_dict.get("edge_triples", []):
        g.add_edge(i, j)
    return g


def _degree_histogram(g: nx.Graph, n_max: int) -> list[float]:
    """Normalized degree histogram, zero-padded to length *n_max*.

    Bin *i* = fraction of nodes with degree *i*. Maximum possible degree
    in an *n_max*-node graph is *n_max* - 1.
    """
    hist = [0.0] * n_max
    n = g.number_of_nodes()
    if n == 0:
        return hist
    for _node, deg in g.degree():
        hist[min(deg, n_max - 1)] += 1
    return [h / n for h in hist]


def _clustering_histogram(g: nx.Graph, n_bins: int = 10) -> list[float]:
    """Histogram of per-node clustering coefficients.

    Values are binned into *n_bins* uniform bins over [0, 1].
    """
    hist = [0.0] * n_bins
    n = g.number_of_nodes()
    if n == 0:
        return [1.0 / n_bins] * n_bins
    cc = nx.clustering(g)
    for val in cc.values():
        idx = min(int(val * n_bins), n_bins - 1)
        hist[idx] += 1
    return [h / n for h in hist]


def _spectral_features(g: nx.Graph, n_max: int) -> list[float]:
    """Eigenvalues of the normalized Laplacian, zero-padded to *n_max*."""
    n = g.number_of_nodes()
    if n == 0:
        return [0.0] * n_max
    if n == 1:
        return [0.0] * n_max
    lap = nx.normalized_laplacian_matrix(g).toarray()
    eigenvalues = sorted(np.linalg.eigvalsh(lap).tolist())
    # Pad or truncate to n_max
    if len(eigenvalues) < n_max:
        eigenvalues.extend([0.0] * (n_max - len(eigenvalues)))
    return eigenvalues[:n_max]


def _compute_mmd(
    x: list[list[float]],
    y: list[list[float]],
    *,
    sigma: float | None = None,
) -> float:
    """Unbiased MMD^2 with Gaussian RBF kernel.

    If *sigma* is ``None``, uses the median heuristic on pairwise
    distances in the combined set.
    """
    if not x or not y:
        return 0.0

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    n, m = len(x_arr), len(y_arr)

    # Pairwise squared distances
    xx = np.sum((x_arr[:, None, :] - x_arr[None, :, :]) ** 2, axis=2)
    yy = np.sum((y_arr[:, None, :] - y_arr[None, :, :]) ** 2, axis=2)
    xy = np.sum((x_arr[:, None, :] - y_arr[None, :, :]) ** 2, axis=2)

    # Median heuristic for sigma
    if sigma is None:
        all_dists = np.concatenate(
            [xx.ravel(), yy.ravel(), xy.ravel()]
        )
        nonzero = all_dists[all_dists > 0]
        if len(nonzero) == 0:
            return 0.0
        sigma = float(np.sqrt(np.median(nonzero) / 2.0))
        if sigma == 0.0:
            return 0.0

    gamma = 1.0 / (2.0 * sigma * sigma)

    k_xx = np.exp(-gamma * xx)
    k_yy = np.exp(-gamma * yy)
    k_xy = np.exp(-gamma * xy)

    # Unbiased estimator (exclude diagonal for k_xx and k_yy)
    if n > 1:
        mmd_xx = (k_xx.sum() - n) / (n * (n - 1))
    else:
        mmd_xx = 0.0
    if m > 1:
        mmd_yy = (k_yy.sum() - m) / (m * (m - 1))
    else:
        mmd_yy = 0.0
    mmd_xy = k_xy.sum() / (n * m)

    mmd2 = mmd_xx + mmd_yy - 2.0 * mmd_xy
    return max(0.0, float(mmd2))
