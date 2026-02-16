"""Graph validity checker for generated bubble diagrams.

Checks structural correctness of generated token sequences: connectivity,
spatial relationship consistency, room-type constraints, and absence of
MASK tokens. Each check produces a boolean flag; ``overall`` is the AND
of all individual checks.

Note on ``vocab_config`` parameter: the spec signature omits it, but
``detokenize`` requires it. Same precedent as Phase 3 (forward_mask,
ELBOLoss).
"""

from __future__ import annotations

from collections import deque

from torch import Tensor

from bd_gen.data.tokenizer import detokenize
from bd_gen.data.vocab import (
    EDGE_MASK_IDX,
    EDGE_NO_EDGE_IDX,
    EDGE_TYPES,
    NODE_MASK_IDX,
    NODE_TYPES,
    VocabConfig,
)


def check_validity(
    tokens: Tensor,
    pad_mask: Tensor,
    vocab_config: VocabConfig,
) -> dict:
    """Check whether a generated token sequence forms a valid bubble diagram.

    Args:
        tokens: ``(seq_len,)`` long tensor of token indices.
        pad_mask: ``(seq_len,)`` bool tensor (True = real position).
        vocab_config: VocabConfig for this dataset.

    Returns:
        Dict with boolean flags for each check, diagnostic info, and
        an ``overall`` flag that is True only if all checks pass.
    """
    n_max = vocab_config.n_max
    details: dict = {}

    # --- Basic sanity: no MASK tokens in real positions ---
    node_real = pad_mask[:n_max]
    edge_real = pad_mask[n_max:]

    node_tokens = tokens[:n_max]
    edge_tokens = tokens[n_max:]

    node_has_mask = bool((node_tokens[node_real] == NODE_MASK_IDX).any().item())
    edge_has_mask = bool((edge_tokens[edge_real] == EDGE_MASK_IDX).any().item())
    no_mask_tokens = not node_has_mask and not edge_has_mask
    if not no_mask_tokens:
        n_mask = (node_tokens[node_real] == NODE_MASK_IDX).sum()
        e_mask = (edge_tokens[edge_real] == EDGE_MASK_IDX).sum()
        details["mask_positions"] = {
            "node_mask_count": int(n_mask.item()),
            "edge_mask_count": int(e_mask.item()),
        }

    # --- Range check: real tokens within valid ranges ---
    if node_real.any():
        nt = node_tokens[node_real]
        node_in_range = bool(((nt >= 0) & (nt < len(NODE_TYPES))).all())
    else:
        node_in_range = True

    if edge_real.any():
        et = edge_tokens[edge_real]
        edge_in_range = bool(((et >= 0) & (et <= EDGE_NO_EDGE_IDX)).all())
    else:
        edge_in_range = True

    no_out_of_range = node_in_range and edge_in_range

    # --- Attempt detokenize (requires no MASK and valid ranges) ---
    num_rooms = int(node_real.sum().item())

    if not (no_mask_tokens and no_out_of_range):
        # Cannot reliably detokenize; skip structural checks
        return {
            "connected": False,
            "consistent": False,
            "valid_types": False,
            "no_mask_tokens": no_mask_tokens,
            "no_out_of_range": no_out_of_range,
            "overall": False,
            "num_rooms": num_rooms,
            "num_edges": 0,
            "details": details,
        }

    try:
        graph_dict = detokenize(tokens, pad_mask, vocab_config)
    except ValueError as exc:
        details["detokenize_error"] = str(exc)
        return {
            "connected": False,
            "consistent": False,
            "valid_types": False,
            "no_mask_tokens": no_mask_tokens,
            "no_out_of_range": no_out_of_range,
            "overall": False,
            "num_rooms": num_rooms,
            "num_edges": 0,
            "details": details,
        }

    num_rooms = graph_dict["num_rooms"]
    edge_triples = graph_dict["edge_triples"]
    node_types = graph_dict["node_types"]

    # --- Connectivity check (BFS) ---
    connected = _check_connected(num_rooms, edge_triples)

    # --- Spatial consistency ---
    # Upper-triangle format stores each pair (i,j) once with i<j, so
    # directional contradictions (e.g. both "left-of" and "right-of" for
    # the same pair) cannot occur. We verify all edge types are valid
    # spatial relationships (already ensured by range check + detokenize,
    # but included for completeness).
    consistent = all(0 <= rel < len(EDGE_TYPES) for _, _, rel in edge_triples)

    # --- Room-type constraints ---
    valid_types, type_details = _check_room_types(node_types)
    if type_details:
        details["type_violations"] = type_details

    overall = (
        connected
        and consistent
        and valid_types
        and no_mask_tokens
        and no_out_of_range
    )

    return {
        "connected": connected,
        "consistent": consistent,
        "valid_types": valid_types,
        "no_mask_tokens": no_mask_tokens,
        "no_out_of_range": no_out_of_range,
        "overall": overall,
        "num_rooms": num_rooms,
        "num_edges": len(edge_triples),
        "details": details,
    }


def check_validity_batch(
    tokens: Tensor,
    pad_mask: Tensor,
    vocab_config: VocabConfig,
) -> list[dict]:
    """Check validity for a batch of samples.

    Args:
        tokens: ``(B, seq_len)`` long tensor.
        pad_mask: ``(B, seq_len)`` bool tensor.
        vocab_config: VocabConfig for this dataset.

    Returns:
        List of validity result dicts, one per sample.
    """
    return [
        check_validity(tokens[i], pad_mask[i], vocab_config)
        for i in range(tokens.size(0))
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_connected(num_rooms: int, edge_triples: list[tuple[int, int, int]]) -> bool:
    """BFS connectivity check on the adjacency graph.

    An edge triple (i, j, rel) contributes an undirected edge between
    nodes i and j (no-edge triples are already filtered by detokenize).
    A single-room graph is trivially connected.
    """
    if num_rooms <= 1:
        return True

    # Build adjacency list
    adj: list[list[int]] = [[] for _ in range(num_rooms)]
    for i, j, _rel in edge_triples:
        adj[i].append(j)
        adj[j].append(i)

    # BFS from node 0
    visited = set()
    queue = deque([0])
    visited.add(0)

    while queue:
        node = queue.popleft()
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == num_rooms


def _check_room_types(node_types: list[int]) -> tuple[bool, dict]:
    """Check room-type constraints.

    Constraints (based on RPLAN domain knowledge):
    - At most 1 LivingRoom (index 0)
    - At most 1 Entrance (index 10)

    Returns:
        (valid, details_dict) where details_dict is non-empty if invalid.
    """
    from collections import Counter

    counts = Counter(node_types)
    violations: dict = {}

    # LivingRoom: at most 1
    if counts.get(0, 0) > 1:
        violations["LivingRoom"] = f"count={counts[0]}, expected at most 1"

    # Entrance: at most 1
    if counts.get(10, 0) > 1:
        violations["Entrance"] = f"count={counts[10]}, expected at most 1"

    return len(violations) == 0, violations
