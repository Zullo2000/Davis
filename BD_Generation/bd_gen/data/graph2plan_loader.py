"""Graph2Plan .mat dataset loader.

Parses the Graph2Plan data.mat file into a list of graph dictionaries,
each containing node types, edge triples, and room counts. Caches the
parsed result as a .pt file for fast subsequent loads.

Data format (from Graph2Plan):
    Each record has .rType (int32 array of room type indices, 0-based)
    and .rEdge (Nx3 int32 array of [u, v, rel_type], 0-based).
    Edges are already upper-triangle (u < v) in the dataset, but we
    filter self-loops (u == v) defensively.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from bd_gen.data.vocab import EDGE_TYPES, NODE_TYPES

logger = logging.getLogger(__name__)

# Number of valid data-bearing edge relationship types (0..9).
_NUM_EDGE_RELATIONS: int = len(EDGE_TYPES)


def _invert_relationship(r: int) -> int:
    """Return the inverse spatial relationship index.

    The Graph2Plan edge vocabulary has 10 relationship types (0-9) where
    each relationship ``r`` has an inverse ``9 - r``:

        0 (left-above)  <-> 9 (right-below)
        1 (left-below)  <-> 8 (right-above)
        2 (left-of)     <-> 7 (right-of)
        3 (above)       <-> 6 (below)
        4 (inside)      <-> 5 (surrounding)

    Args:
        r: Relationship index in [0, 9].

    Returns:
        Inverse relationship index in [0, 9].

    Raises:
        ValueError: If *r* is outside the valid range.
    """
    if not (0 <= r < _NUM_EDGE_RELATIONS):
        raise ValueError(
            f"Relationship index must be in [0, {_NUM_EDGE_RELATIONS - 1}], "
            f"got {r}"
        )
    return 9 - r


def _parse_record(record: object) -> dict | None:
    """Parse a single Graph2Plan record into a graph dict.

    Args:
        record: A single element from the loaded .mat struct array.
                Expected attributes: ``.rType`` and ``.rEdge``.

    Returns:
        A dict with keys ``"node_types"``, ``"edge_triples"``, and
        ``"num_rooms"``, or ``None`` if the record is degenerate (e.g.
        zero rooms).
    """
    # --- Node types ---
    rtype_raw = np.asarray(record.rType, dtype=np.int32)

    # squeeze_me=True can reduce a single-element array to a scalar
    if rtype_raw.ndim == 0:
        rtype_raw = rtype_raw.reshape(1)

    node_types: list[int] = rtype_raw.tolist()
    num_rooms = len(node_types)

    if num_rooms == 0:
        return None

    # --- Edge triples ---
    redge_raw = np.asarray(record.rEdge, dtype=np.int32)

    edge_triples: list[tuple[int, int, int]] = []

    if redge_raw.size == 0:
        # No edges (possible for single-room graphs)
        pass
    else:
        # Ensure 2-D shape (Nx3); a single edge may be squeezed to 1-D
        if redge_raw.ndim == 1:
            redge_raw = redge_raw.reshape(1, -1)

        assert redge_raw.shape[1] == 3, (
            f"Expected Nx3 edge array, got shape {redge_raw.shape}"
        )

        for row in redge_raw:
            u, v, rel = int(row[0]), int(row[1]), int(row[2])

            # Filter self-loops
            if u == v:
                continue

            # Ensure upper-triangle ordering (u < v).
            # Data is already upper-triangle, but be defensive.
            if u > v:
                u, v = v, u
                rel = _invert_relationship(rel)

            edge_triples.append((u, v, rel))

    return {
        "node_types": node_types,
        "edge_triples": edge_triples,
        "num_rooms": num_rooms,
    }


def load_graph2plan(
    mat_path: str | Path,
    cache_path: str | Path,
    n_max: int = 8,
) -> list[dict]:
    """Load and parse the Graph2Plan dataset.

    If *cache_path* exists, the cached list is loaded directly.
    Otherwise, *mat_path* is parsed with ``scipy.io.loadmat``, filtered,
    validated, and saved to *cache_path* for future use.

    Args:
        mat_path: Path to the ``data.mat`` file from Graph2Plan.
        cache_path: Path for the ``.pt`` cache file.
        n_max: Maximum number of rooms per graph. Graphs with more rooms
               are silently skipped (count is logged).

    Returns:
        List of graph dicts, each containing:
            - ``"node_types"``: ``list[int]`` of room-type indices (0-12).
            - ``"edge_triples"``: ``list[tuple[int, int, int]]`` of
              ``(u, v, rel_type)`` with ``u < v``, all 0-based.
            - ``"num_rooms"``: ``int``, equal to ``len(node_types)``.

    Raises:
        FileNotFoundError: If *mat_path* does not exist and no cache.
        AssertionError: If any parsed values fall outside valid ranges.
    """
    mat_path = Path(mat_path)
    cache_path = Path(cache_path)

    # --- Try cache first ---
    if cache_path.exists():
        logger.info("Loading cached dataset from %s", cache_path)
        graphs: list[dict] = torch.load(cache_path, weights_only=False)
        logger.info("Loaded %d graphs from cache", len(graphs))
        return graphs

    # --- Parse from .mat ---
    if not mat_path.exists():
        raise FileNotFoundError(
            f"data.mat not found at {mat_path}. "
            "Run scripts/prepare_data.py to download it."
        )

    logger.info("Parsing %s ...", mat_path)
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    records = mat["data"]  # ndarray of shape (N,)

    graphs = []
    skipped_nmax = 0
    skipped_degenerate = 0

    for rec in records:
        graph = _parse_record(rec)

        if graph is None:
            skipped_degenerate += 1
            continue

        if graph["num_rooms"] > n_max:
            skipped_nmax += 1
            continue

        # --- Validate ranges ---
        for nt in graph["node_types"]:
            assert 0 <= nt < len(NODE_TYPES), (
                f"Invalid node type {nt}, expected [0, {len(NODE_TYPES) - 1}]"
            )

        for u, v, rel in graph["edge_triples"]:
            assert 0 <= u < v < graph["num_rooms"], (
                f"Invalid edge pair ({u}, {v}) for num_rooms={graph['num_rooms']}"
            )
            assert 0 <= rel < len(EDGE_TYPES), (
                f"Invalid edge type {rel}, expected [0, {len(EDGE_TYPES) - 1}]"
            )

        graphs.append(graph)

    logger.info(
        "Parsed %d graphs (%d skipped: %d exceeded n_max=%d, %d degenerate)",
        len(graphs),
        skipped_nmax + skipped_degenerate,
        skipped_nmax,
        n_max,
        skipped_degenerate,
    )

    # --- Cache ---
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, cache_path)
    logger.info("Saved cache to %s", cache_path)

    return graphs
