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
from collections import Counter, defaultdict

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
    zeros = {
        "node_kl": 0.0, "edge_kl": 0.0, "num_rooms_kl": 0.0,
        "node_js": 0.0, "edge_js": 0.0,
        "node_tv": 0.0, "edge_tv": 0.0,
        "rooms_w1": 0.0,
    }
    if not samples or not training_set:
        return zeros

    # Node type histograms
    sample_node_hist = _node_histogram(samples)
    train_node_hist = _node_histogram(training_set)
    node_kl = _kl_divergence(sample_node_hist, train_node_hist)
    node_js = _js_divergence(sample_node_hist, train_node_hist)
    node_tv = _total_variation(sample_node_hist, train_node_hist)

    # Edge type histograms (spatial relationships only, indices 0-9)
    sample_edge_hist = _edge_histogram(samples)
    train_edge_hist = _edge_histogram(training_set)
    edge_kl = _kl_divergence(sample_edge_hist, train_edge_hist)
    edge_js = _js_divergence(sample_edge_hist, train_edge_hist)
    edge_tv = _total_variation(sample_edge_hist, train_edge_hist)

    # Num rooms histograms
    sample_rooms_hist = _num_rooms_histogram(samples)
    train_rooms_hist = _num_rooms_histogram(training_set)
    num_rooms_kl = _kl_divergence(sample_rooms_hist, train_rooms_hist)
    rooms_w1 = _wasserstein1_1d_discrete(sample_rooms_hist, train_rooms_hist)

    return {
        "node_kl": node_kl,
        "edge_kl": edge_kl,
        "num_rooms_kl": num_rooms_kl,
        "node_js": node_js,
        "edge_js": edge_js,
        "node_tv": node_tv,
        "edge_tv": edge_tv,
        "rooms_w1": rooms_w1,
    }


def conditional_edge_kl(
    samples: list[dict],
    training_set: list[dict],
    *,
    min_pair_count: int = 5,
) -> dict[str, float]:
    """KL divergence of edge types conditioned on canonical room-type pair.

    Unlike the marginal ``edge_kl`` in :func:`distribution_match`, this
    computes the edge-type distribution separately for each room-type pair
    (e.g. LivingRoom–Bathroom) and then averages the per-pair KL values.
    This catches cases where the model applies the wrong spatial
    relationship for a specific room combination (e.g. "Bath surrounding
    LivingRoom").

    Room-type pairs are **canonicalized**: if ``type_i > type_j``, the
    pair is swapped and ``rel`` is inverted (``9 - rel``), so the same
    physical arrangement always maps to the same ``(pair, rel)``
    regardless of node ordering.

    Args:
        samples: Generated graph dicts.
        training_set: Training graph dicts.
        min_pair_count: Skip training pairs with fewer total edges than
            this.  Sparse pairs have unreliable histograms.

    Returns:
        Dict with ``conditional_edge_kl_mean`` (unweighted mean KL over
        eligible pairs), ``conditional_edge_kl_weighted`` (training-
        frequency-weighted mean KL), and ``num_pairs_evaluated`` (number
        of canonical pairs compared).  Returns all zeros if either input
        is empty or no pairs pass the threshold.
    """
    zeros = {
        "conditional_edge_kl_mean": 0.0,
        "conditional_edge_kl_weighted": 0.0,
        "conditional_edge_js_mean": 0.0,
        "conditional_edge_js_weighted": 0.0,
        "conditional_edge_tv_mean": 0.0,
        "conditional_edge_tv_weighted": 0.0,
        "num_pairs_evaluated": 0.0,
    }
    if not samples or not training_set:
        return zeros

    train_hists, train_counts = _conditional_edge_histogram(training_set)
    sample_hists, _ = _conditional_edge_histogram(samples)

    # Filter to pairs with enough training data
    eligible_pairs = [
        pair for pair, count in train_counts.items()
        if count >= min_pair_count
    ]
    if not eligible_pairs:
        return zeros

    total_train_edges = sum(train_counts[p] for p in eligible_pairs)
    n_rel = len(EDGE_TYPES)
    uniform = [1.0 / n_rel] * n_rel

    kl_values: list[float] = []
    js_values: list[float] = []
    tv_values: list[float] = []
    weights: list[float] = []

    for pair in eligible_pairs:
        train_hist = train_hists[pair]
        # Uniform fallback when pair has no sample edges (avoids false KL=0)
        sample_hist = sample_hists.get(pair, uniform)

        kl_values.append(_kl_divergence(sample_hist, train_hist))
        js_values.append(_js_divergence(sample_hist, train_hist))
        tv_values.append(_total_variation(sample_hist, train_hist))
        weights.append(train_counts[pair] / total_train_edges)

    n = len(kl_values)
    return {
        "conditional_edge_kl_mean": sum(kl_values) / n,
        "conditional_edge_kl_weighted": sum(
            w * v for w, v in zip(weights, kl_values)
        ),
        "conditional_edge_js_mean": sum(js_values) / n,
        "conditional_edge_js_weighted": sum(
            w * v for w, v in zip(weights, js_values)
        ),
        "conditional_edge_tv_mean": sum(tv_values) / n,
        "conditional_edge_tv_weighted": sum(
            w * v for w, v in zip(weights, tv_values)
        ),
        "num_pairs_evaluated": float(n),
    }


def conditional_edge_distances_topN(
    samples: list[dict],
    training_set: list[dict],
    *,
    top_n: int = 20,
    min_pair_count: int = 5,
) -> dict[str, float]:
    """KL/JS/TV of edge types for the top-N most frequent canonical pairs.

    Selects the *top_n* most frequent canonical room-type pairs in the
    **training set** (by edge count) and computes per-pair KL, JS, and TV.
    Using only high-frequency pairs gives more stable estimates across runs.

    Args:
        samples: Generated graph dicts.
        training_set: Training graph dicts.
        top_n: Number of top pairs to evaluate.
        min_pair_count: Minimum training edges per pair (applied before
            top-N selection).

    Returns:
        Dict with mean/weighted KL/JS/TV, ``topN`` used, and
        ``num_pairs_evaluated``.
    """
    zeros = {
        "conditional_edge_kl_topN_mean": 0.0,
        "conditional_edge_kl_topN_weighted": 0.0,
        "conditional_edge_js_topN_mean": 0.0,
        "conditional_edge_js_topN_weighted": 0.0,
        "conditional_edge_tv_topN_mean": 0.0,
        "conditional_edge_tv_topN_weighted": 0.0,
        "topN": float(top_n),
        "num_pairs_evaluated": 0.0,
    }
    if not samples or not training_set:
        return zeros

    train_hists, train_counts = _conditional_edge_histogram(training_set)
    sample_hists, _ = _conditional_edge_histogram(samples)

    eligible = [
        (pair, count) for pair, count in train_counts.items()
        if count >= min_pair_count
    ]
    eligible.sort(key=lambda x: x[1], reverse=True)
    top_pairs = [pair for pair, _ in eligible[:top_n]]
    if not top_pairs:
        return zeros

    total_train_edges = sum(train_counts[p] for p in top_pairs)
    n_rel = len(EDGE_TYPES)
    uniform = [1.0 / n_rel] * n_rel

    kl_values: list[float] = []
    js_values: list[float] = []
    tv_values: list[float] = []
    weights: list[float] = []

    for pair in top_pairs:
        train_hist = train_hists[pair]
        sample_hist = sample_hists.get(pair, uniform)
        kl_values.append(_kl_divergence(sample_hist, train_hist))
        js_values.append(_js_divergence(sample_hist, train_hist))
        tv_values.append(_total_variation(sample_hist, train_hist))
        weights.append(train_counts[pair] / total_train_edges)

    n = len(kl_values)
    return {
        "conditional_edge_kl_topN_mean": sum(kl_values) / n,
        "conditional_edge_kl_topN_weighted": sum(
            w * v for w, v in zip(weights, kl_values)
        ),
        "conditional_edge_js_topN_mean": sum(js_values) / n,
        "conditional_edge_js_topN_weighted": sum(
            w * v for w, v in zip(weights, js_values)
        ),
        "conditional_edge_tv_topN_mean": sum(tv_values) / n,
        "conditional_edge_tv_topN_weighted": sum(
            w * v for w, v in zip(weights, tv_values)
        ),
        "topN": float(top_n),
        "num_pairs_evaluated": float(n),
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
    max_samples: int = 5000,
    seed: int = 42,
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
        max_samples: Maximum number of graphs from each set to use. Both
            sets are randomly subsampled (without replacement) if they
            exceed this limit. Prevents OOM on large datasets.
        seed: Random seed for reproducible subsampling.

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

    # Subsample to avoid OOM on large datasets
    rng = np.random.RandomState(seed)
    if len(sample_graphs) > max_samples:
        idx = rng.choice(len(sample_graphs), max_samples, replace=False)
        sample_graphs = [sample_graphs[i] for i in idx]
    if len(ref_graphs) > max_samples:
        idx = rng.choice(len(ref_graphs), max_samples, replace=False)
        ref_graphs = [ref_graphs[i] for i in idx]

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


def spatial_transitivity(graph_dicts: list[dict]) -> dict[str, float]:
    """Fraction of graphs whose spatial relationships are 2D-consistent.

    A bubble diagram can pass all validity checks (connected, correct
    types, no MASK residue) yet be *physically impossible* because its
    spatial relationships contradict each other.  For example, if room A
    is left-of B and B is left-of C, then A cannot be right-of C — that
    would require A to be both left *and* right of C in the same layout.

    This metric decomposes each edge relationship into horizontal and
    vertical ordering constraints and checks for directed cycles on each
    axis.  A cycle means no 2D placement can satisfy all relationships
    simultaneously.

    Relationships ``inside`` (4) and ``surrounding`` (5) are containment
    relationships that do not impose strict positional ordering, so they
    contribute no constraints.

    Args:
        graph_dicts: List of detokenized graph dicts.

    Returns:
        Dict with ``transitivity_score`` (fraction with no contradictions,
        higher is better), ``h_consistent`` (no horizontal contradictions),
        and ``v_consistent`` (no vertical contradictions).
        Returns all 1.0 for empty input.
    """
    if not graph_dicts:
        return {
            "transitivity_score": 1.0,
            "h_consistent": 1.0,
            "v_consistent": 1.0,
        }

    h_ok = 0
    v_ok = 0
    both_ok = 0
    for g in graph_dicts:
        result = _check_spatial_consistency(g)
        if result["h_consistent"]:
            h_ok += 1
        if result["v_consistent"]:
            v_ok += 1
        if result["overall"]:
            both_ok += 1

    n = len(graph_dicts)
    return {
        "transitivity_score": both_ok / n,
        "h_consistent": h_ok / n,
        "v_consistent": v_ok / n,
    }


def type_conditioned_degree_kl(
    samples: list[dict],
    reference: list[dict],
    *,
    n_max: int = 8,
    min_type_count: int = 20,
) -> dict[str, float]:
    """KL divergence of node degree distributions conditioned on room type.

    In real floorplans, different room types have characteristic
    connectivity patterns: bathrooms typically connect to 1--2 rooms while
    living rooms connect to 3--5.  The global ``mmd_degree`` metric pools
    all room types into a single histogram, so a model could match the
    overall degree distribution while getting per-type connectivity wrong.

    This metric computes, for each room type, a histogram of node degrees
    across all graphs, then measures KL(samples || reference) per type.

    Room types with fewer than *min_type_count* nodes in the reference set
    are excluded (their empirical histogram is too noisy).

    Args:
        samples: Generated graph dicts.
        reference: Reference (training) graph dicts.
        n_max: Maximum number of rooms (histogram length).
        min_type_count: Minimum reference nodes to include a type.

    Returns:
        Dict with ``degree_kl_per_type_mean`` (unweighted mean KL),
        ``degree_kl_per_type_weighted`` (reference-frequency-weighted
        mean KL), and ``num_types_evaluated``.
        Returns all zeros if either input is empty or no types qualify.
    """
    zeros = {
        "degree_kl_per_type_mean": 0.0,
        "degree_kl_per_type_weighted": 0.0,
        "degree_js_per_type_mean": 0.0,
        "degree_js_per_type_weighted": 0.0,
        "degree_tv_per_type_mean": 0.0,
        "degree_tv_per_type_weighted": 0.0,
        "num_types_evaluated": 0.0,
    }
    if not samples or not reference:
        return zeros

    ref_hists, ref_counts = _per_type_degree_histograms(reference, n_max)
    sample_hists, _ = _per_type_degree_histograms(samples, n_max)

    eligible_types = [
        t for t, c in ref_counts.items() if c >= min_type_count
    ]
    if not eligible_types:
        return zeros

    total_ref_nodes = sum(ref_counts[t] for t in eligible_types)
    uniform = [1.0 / n_max] * n_max

    kl_values: list[float] = []
    js_values: list[float] = []
    tv_values: list[float] = []
    weights: list[float] = []

    for room_type in eligible_types:
        ref_hist = ref_hists[room_type]
        sample_hist = sample_hists.get(room_type, uniform)
        kl_values.append(_kl_divergence(sample_hist, ref_hist))
        js_values.append(_js_divergence(sample_hist, ref_hist))
        tv_values.append(_total_variation(sample_hist, ref_hist))
        weights.append(ref_counts[room_type] / total_ref_nodes)

    n = len(kl_values)
    return {
        "degree_kl_per_type_mean": sum(kl_values) / n,
        "degree_kl_per_type_weighted": sum(
            w * v for w, v in zip(weights, kl_values)
        ),
        "degree_js_per_type_mean": sum(js_values) / n,
        "degree_js_per_type_weighted": sum(
            w * v for w, v in zip(weights, js_values)
        ),
        "degree_tv_per_type_mean": sum(tv_values) / n,
        "degree_tv_per_type_weighted": sum(
            w * v for w, v in zip(weights, tv_values)
        ),
        "num_types_evaluated": float(n),
    }


def mode_coverage(
    samples: list[dict],
    training_set: list[dict],
) -> dict[str, float]:
    """Fraction of training room-type combinations covered by samples.

    ``diversity`` measures whether generated samples differ from *each
    other*; ``novelty`` measures whether they differ from *training data*.
    Neither checks whether the model covers the full **range** of graph
    archetypes.  A model that generates 1000 unique 4-bedroom layouts but
    never produces a studio apartment scores high on both diversity and
    novelty, yet suffers from mode collapse.

    An "archetype" is the sorted multiset of room types in a graph (e.g.
    ``(LivingRoom, Kitchen, Bathroom, Bedroom)``).  This metric computes
    what fraction of training archetypes appear at least once in the
    generated set.

    Args:
        samples: Generated graph dicts.
        training_set: Training graph dicts.

    Returns:
        Dict with ``mode_coverage`` (unweighted: fraction of distinct
        training archetypes produced), ``mode_coverage_weighted``
        (weighted by training archetype frequency), ``num_training_modes``
        (total distinct archetypes in training), and ``num_sample_modes``
        (total distinct archetypes in samples).
        Returns all zeros if either input is empty.
    """
    if not samples or not training_set:
        return {
            "mode_coverage": 0.0,
            "mode_coverage_weighted": 0.0,
            "num_training_modes": 0.0,
            "num_sample_modes": 0.0,
        }

    # Training archetype frequencies
    train_counts: Counter = Counter()
    for g in training_set:
        train_counts[_archetype_hash(g)] += 1

    sample_archetypes = {_archetype_hash(g) for g in samples}

    n_train_modes = len(train_counts)
    if n_train_modes == 0:
        return {
            "mode_coverage": 0.0,
            "mode_coverage_weighted": 0.0,
            "num_training_modes": 0.0,
            "num_sample_modes": float(len(sample_archetypes)),
        }

    covered = sample_archetypes & set(train_counts.keys())
    unweighted = len(covered) / n_train_modes

    total_train = sum(train_counts.values())
    weighted = sum(train_counts[a] for a in covered) / total_train

    return {
        "mode_coverage": unweighted,
        "mode_coverage_weighted": weighted,
        "num_training_modes": float(n_train_modes),
        "num_sample_modes": float(len(sample_archetypes)),
    }


def validity_by_num_rooms(
    validity_results: list[dict],
    graph_dicts: list[dict],
) -> dict[str, dict[str, float]]:
    """Validity rates stratified by num_rooms.

    Args:
        validity_results: List of dicts from ``check_validity_batch``.
        graph_dicts: Corresponding detokenized graph dicts (same order).

    Returns:
        Dict mapping str(num_rooms) to a dict of check-name -> rate.
    """
    if not validity_results or not graph_dicts:
        return {}

    groups: dict[int, list[dict]] = defaultdict(list)
    for vr, gd in zip(validity_results, graph_dicts):
        nr = gd.get("num_rooms", 0)
        groups[nr].append(vr)

    result: dict[str, dict[str, float]] = {}
    for nr in sorted(groups):
        items = groups[nr]
        n = len(items)
        checks: dict[str, float] = {}
        # Get all boolean keys from first item
        for key in items[0]:
            if isinstance(items[0][key], bool):
                checks[key] = sum(1 for r in items if r[key]) / n
        result[str(nr)] = checks

    return result


def spatial_transitivity_by_num_rooms(
    graph_dicts: list[dict],
) -> dict[str, dict[str, float]]:
    """Spatial transitivity stratified by num_rooms.

    Groups graphs by ``num_rooms`` and runs :func:`spatial_transitivity`
    on each group.

    Returns:
        Dict mapping str(num_rooms) to transitivity result dict.
    """
    if not graph_dicts:
        return {}

    groups: dict[int, list[dict]] = defaultdict(list)
    for gd in graph_dicts:
        groups[gd.get("num_rooms", 0)].append(gd)

    result: dict[str, dict[str, float]] = {}
    for nr in sorted(groups):
        result[str(nr)] = spatial_transitivity(groups[nr])

    return result


def edge_present_rate_by_num_rooms(
    graph_dicts: list[dict],
) -> dict[str, float]:
    """Mean edge-present rate stratified by num_rooms.

    Edge-present rate per graph = E_present / E_possible, where
    E_possible = n*(n-1)/2 and E_present = len(edge_triples).

    Returns:
        Dict mapping str(num_rooms) to mean edge-present rate.
    """
    if not graph_dicts:
        return {}

    groups: dict[int, list[float]] = defaultdict(list)
    for gd in graph_dicts:
        n = gd.get("num_rooms", 0)
        e_possible = n * (n - 1) / 2
        e_present = len(gd.get("edge_triples", []))
        rate = e_present / e_possible if e_possible > 0 else 0.0
        groups[n].append(rate)

    result: dict[str, float] = {}
    for nr in sorted(groups):
        rates = groups[nr]
        result[str(nr)] = sum(rates) / len(rates)

    return result


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


def _total_variation(p: list[float], q: list[float]) -> float:
    """Total Variation distance: TV(p, q) = 0.5 * sum_k |p_k - q_k|.

    Handles different-length inputs by zero-padding the shorter one.
    """
    max_len = max(len(p), len(q))
    tv = 0.0
    for i in range(max_len):
        pi = p[i] if i < len(p) else 0.0
        qi = q[i] if i < len(q) else 0.0
        tv += abs(pi - qi)
    return 0.5 * tv


def _js_divergence(p: list[float], q: list[float]) -> float:
    """Jensen-Shannon divergence (in nats).

    JS(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m) where m = 0.5*(p+q).

    No epsilon smoothing needed: if p_k > 0 then m_k >= 0.5*p_k > 0,
    so log(p_k / m_k) is always safe.  Convention: 0 * log(0/x) = 0.

    Handles different-length inputs by zero-padding the shorter one.
    """
    max_len = max(len(p), len(q))
    kl_pm = 0.0
    kl_qm = 0.0
    for i in range(max_len):
        pi = p[i] if i < len(p) else 0.0
        qi = q[i] if i < len(q) else 0.0
        mi = 0.5 * (pi + qi)
        if pi > 0 and mi > 0:
            kl_pm += pi * math.log(pi / mi)
        if qi > 0 and mi > 0:
            kl_qm += qi * math.log(qi / mi)
    return 0.5 * kl_pm + 0.5 * kl_qm


def _wasserstein1_1d_discrete(p: list[float], q: list[float]) -> float:
    """Wasserstein-1 distance for 1D discrete distributions with unit spacing.

    W1(p, q) = sum_k |CDF_p(k) - CDF_q(k)|.

    Meaningful for ordinal variables (e.g. num_rooms) where being
    "one bin off" should cost less than "four bins off".

    Handles different-length inputs by zero-padding the shorter one.
    """
    max_len = max(len(p), len(q))
    cdf_p = 0.0
    cdf_q = 0.0
    w1 = 0.0
    for i in range(max_len):
        cdf_p += p[i] if i < len(p) else 0.0
        cdf_q += q[i] if i < len(q) else 0.0
        w1 += abs(cdf_p - cdf_q)
    return w1


def _canonicalize_edge(
    type_i: int, type_j: int, rel: int,
) -> tuple[tuple[int, int], int]:
    """Canonical form for a room-type-pair + relationship.

    Ensures the lower room-type index comes first.  If swapped, the
    relationship is inverted (``9 - rel``) to preserve spatial semantics.
    """
    if type_i > type_j:
        return (type_j, type_i), 9 - rel
    return (type_i, type_j), rel


def _conditional_edge_histogram(
    graph_dicts: list[dict],
) -> tuple[dict[tuple[int, int], list[float]], dict[tuple[int, int], int]]:
    """Per-canonical-room-pair edge-type histograms.

    Returns:
        A tuple ``(histograms, counts)`` where *histograms* maps each
        canonical pair to a normalized ``list[float]`` of length
        ``len(EDGE_TYPES)`` and *counts* maps each pair to its raw
        total edge count.
    """
    raw: dict[tuple[int, int], Counter] = defaultdict(Counter)

    for g in graph_dicts:
        node_types = g["node_types"]
        for node_i, node_j, rel in g["edge_triples"]:
            type_i = node_types[node_i]
            type_j = node_types[node_j]
            pair, canon_rel = _canonicalize_edge(type_i, type_j, rel)
            raw[pair][canon_rel] += 1

    n_rel = len(EDGE_TYPES)
    histograms: dict[tuple[int, int], list[float]] = {}
    counts: dict[tuple[int, int], int] = {}

    for pair, counter in raw.items():
        total = sum(counter.values())
        counts[pair] = total
        histograms[pair] = [counter.get(i, 0) / total for i in range(n_rel)]

    return histograms, counts


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


# ---------------------------------------------------------------------------
# Spatial transitivity helpers
# ---------------------------------------------------------------------------

# Horizontal and vertical ordering constraints per edge type.
# "left" means A.x < B.x, "right" means A.x > B.x (H-axis).
# "above" means A.y > B.y, "below" means A.y < B.y (V-axis).
# None = no constraint on that axis.
# inside/surrounding are containment — no strict positional ordering.
_H_CONSTRAINT: dict[int, str | None] = {
    0: "left",      # left-above
    1: "left",      # left-below
    2: "left",      # left-of
    3: None,        # above
    4: None,        # inside
    5: None,        # surrounding
    6: None,        # below
    7: "right",     # right-of
    8: "right",     # right-above
    9: "right",     # right-below
}
_V_CONSTRAINT: dict[int, str | None] = {
    0: "above",     # left-above
    1: "below",     # left-below
    2: None,        # left-of
    3: "above",     # above
    4: None,        # inside
    5: None,        # surrounding
    6: "below",     # below
    7: None,        # right-of
    8: "above",     # right-above
    9: "below",     # right-below
}


def _has_cycle(adj: list[list[int]], n: int) -> bool:
    """Detect a cycle in a directed graph via DFS.

    Args:
        adj: Adjacency list — adj[u] is the list of nodes reachable from u.
        n: Number of nodes.

    Returns:
        True if the directed graph contains a cycle.
    """
    # 0 = unvisited, 1 = in current DFS stack, 2 = finished
    state = [0] * n
    for start in range(n):
        if state[start] == 2:
            continue
        stack = [(start, 0)]
        state[start] = 1
        while stack:
            node, idx = stack[-1]
            if idx < len(adj[node]):
                stack[-1] = (node, idx + 1)
                nxt = adj[node][idx]
                if state[nxt] == 1:
                    return True
                if state[nxt] == 0:
                    state[nxt] = 1
                    stack.append((nxt, 0))
            else:
                state[node] = 2
                stack.pop()
    return False


def _check_spatial_consistency(graph_dict: dict) -> dict[str, bool]:
    """Check one graph for spatial contradictions on H and V axes.

    Builds directed ordering graphs for horizontal and vertical axes
    from edge relationships, then checks for cycles.
    """
    n = graph_dict.get("num_rooms", 0)
    if n <= 1:
        return {"h_consistent": True, "v_consistent": True, "overall": True}

    h_adj: list[list[int]] = [[] for _ in range(n)]
    v_adj: list[list[int]] = [[] for _ in range(n)]

    for node_i, node_j, rel in graph_dict.get("edge_triples", []):
        h = _H_CONSTRAINT.get(rel)
        v = _V_CONSTRAINT.get(rel)

        # Horizontal: "left" means i.x < j.x → edge i→j in ordering graph
        # "right" means i.x > j.x → edge j→i
        if h == "left":
            h_adj[node_i].append(node_j)
        elif h == "right":
            h_adj[node_j].append(node_i)

        # Vertical: "above" means i.y > j.y → edge j→i (j is lower)
        # "below" means i.y < j.y → edge i→j
        if v == "above":
            v_adj[node_j].append(node_i)
        elif v == "below":
            v_adj[node_i].append(node_j)

    h_ok = not _has_cycle(h_adj, n)
    v_ok = not _has_cycle(v_adj, n)

    return {"h_consistent": h_ok, "v_consistent": v_ok, "overall": h_ok and v_ok}


# ---------------------------------------------------------------------------
# Type-conditioned degree helpers
# ---------------------------------------------------------------------------


def _per_type_degree_histograms(
    graph_dicts: list[dict],
    n_max: int,
) -> tuple[dict[int, list[float]], dict[int, int]]:
    """Per-room-type degree histograms across a set of graphs.

    Returns:
        ``(histograms, counts)`` where *histograms* maps room type index
        to a normalized degree histogram of length *n_max*, and *counts*
        maps room type to total node count.
    """
    raw: dict[int, Counter] = defaultdict(Counter)
    type_counts: dict[int, int] = defaultdict(int)

    for g in graph_dicts:
        node_types = g["node_types"]
        n = g.get("num_rooms", len(node_types))
        # Build adjacency count per node
        degree = [0] * n
        for node_i, node_j, _rel in g.get("edge_triples", []):
            degree[node_i] += 1
            degree[node_j] += 1

        for idx in range(n):
            room_type = node_types[idx]
            deg = min(degree[idx], n_max - 1)
            raw[room_type][deg] += 1
            type_counts[room_type] += 1

    histograms: dict[int, list[float]] = {}
    for room_type, counter in raw.items():
        total = sum(counter.values())
        histograms[room_type] = [
            counter.get(i, 0) / total for i in range(n_max)
        ]

    return histograms, dict(type_counts)


# ---------------------------------------------------------------------------
# Mode coverage helpers
# ---------------------------------------------------------------------------


def _archetype_hash(graph_dict: dict) -> tuple[int, ...]:
    """Sorted tuple of node types — the room-type "archetype" of a graph."""
    return tuple(sorted(graph_dict["node_types"]))
