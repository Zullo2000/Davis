"""Tokenizer for bubble diagram graphs.

Converts between the intermediate graph dict representation and flat token
sequences suitable for the diffusion model. This is the bridge between
dataset-level graph data and model-level token tensors.

The token sequence layout is:

    [ node_0, node_1, ..., node_{n_max-1}, edge_0, edge_1, ..., edge_{n_edges-1} ]

where each ``edge_k`` corresponds to the upper-triangle pair returned by
``VocabConfig.edge_position_to_pair(k)``.

**Critical invariant**: PAD (position doesn't exist for this graph size) vs
no-edge (both rooms exist but are not adjacent). Real edge positions that
have no corresponding triple get ``EDGE_NO_EDGE_IDX`` (10), *never*
``EDGE_PAD_IDX`` (12).
"""

from __future__ import annotations

import torch
from torch import Tensor

from bd_gen.data.vocab import (
    EDGE_NO_EDGE_IDX,
    EDGE_PAD_IDX,
    EDGE_TYPES,
    NODE_PAD_IDX,
    NODE_TYPES,
    VocabConfig,
)


def tokenize(
    graph_dict: dict,
    vocab_config: VocabConfig,
) -> tuple[Tensor, Tensor]:
    """Convert an intermediate graph dict into a flat token sequence.

    Args:
        graph_dict: Dictionary with keys:
            - ``"num_rooms"`` (int): Number of rooms, 1 <= num_rooms <= n_max.
            - ``"node_types"`` (list[int]): Room type indices, length == num_rooms.
              Each value in ``[0, len(NODE_TYPES))``.
            - ``"edge_triples"`` (list[tuple[int, int, int]]): Each entry is
              ``(i, j, rel)`` where ``0 <= i < j < num_rooms`` and
              ``0 <= rel < len(EDGE_TYPES)``.
        vocab_config: VocabConfig controlling n_max, seq_len, etc.

    Returns:
        Tuple of:
            - tokens: ``Tensor(seq_len,)`` dtype ``torch.long``.
            - pad_mask: ``Tensor(seq_len,)`` dtype ``torch.bool``.
              ``True`` = real position, ``False`` = PAD position.

    Raises:
        ValueError: If any input violates the expected constraints.
    """
    num_rooms: int = graph_dict["num_rooms"]
    node_types: list[int] = graph_dict["node_types"]
    edge_triples: list[tuple[int, int, int]] = graph_dict["edge_triples"]

    # --- Validation ---
    if not (1 <= num_rooms <= vocab_config.n_max):
        raise ValueError(
            f"num_rooms must be in [1, {vocab_config.n_max}], got {num_rooms}"
        )
    if len(node_types) != num_rooms:
        raise ValueError(
            f"len(node_types) must equal num_rooms ({num_rooms}), "
            f"got {len(node_types)}"
        )
    for idx, nt in enumerate(node_types):
        if not (0 <= nt < len(NODE_TYPES)):
            raise ValueError(
                f"node_types[{idx}] = {nt} not in [0, {len(NODE_TYPES)})"
            )
    for triple in edge_triples:
        i, j, rel = triple
        if not (i < j):
            raise ValueError(
                f"edge triple ({i}, {j}, {rel}): requires i < j"
            )
        if not (0 <= i < num_rooms and 0 <= j < num_rooms):
            raise ValueError(
                f"edge triple ({i}, {j}, {rel}): node indices out of range "
                f"[0, {num_rooms})"
            )
        if not (0 <= rel < len(EDGE_TYPES)):
            raise ValueError(
                f"edge triple ({i}, {j}, {rel}): rel not in "
                f"[0, {len(EDGE_TYPES)})"
            )

    # --- Build pad mask ---
    pad_mask = vocab_config.compute_pad_mask(num_rooms)

    # --- Build token sequence ---
    tokens = torch.zeros(vocab_config.seq_len, dtype=torch.long)

    # Fill node positions [0 : n_max]
    for k in range(vocab_config.n_max):
        if k < num_rooms:
            tokens[k] = node_types[k]
        else:
            tokens[k] = NODE_PAD_IDX

    # Fill edge positions [n_max : seq_len]
    for pos in range(vocab_config.n_edges):
        seq_idx = vocab_config.n_max + pos
        if pad_mask[seq_idx]:
            # Real edge position: default to no-edge
            tokens[seq_idx] = EDGE_NO_EDGE_IDX
        else:
            # PAD edge position
            tokens[seq_idx] = EDGE_PAD_IDX

    # Overwrite real edge positions with actual relationships
    for i, j, rel in edge_triples:
        pos = vocab_config.pair_to_edge_position(i, j)
        seq_idx = vocab_config.n_max + pos
        tokens[seq_idx] = rel

    return tokens, pad_mask


def detokenize(
    tokens: Tensor,
    pad_mask: Tensor,
    vocab_config: VocabConfig,
) -> dict:
    """Reconstruct an intermediate graph dict from a token sequence.

    This is the inverse of :func:`tokenize`. Given a valid token sequence
    and its pad mask, it recovers the original graph structure.

    Args:
        tokens: ``Tensor(seq_len,)`` dtype ``torch.long``.
        pad_mask: ``Tensor(seq_len,)`` dtype ``torch.bool``.
            ``True`` = real position, ``False`` = PAD position.
        vocab_config: VocabConfig controlling n_max, seq_len, etc.

    Returns:
        Dictionary with keys:
            - ``"num_rooms"`` (int): Number of rooms.
            - ``"node_types"`` (list[int]): Room type indices.
            - ``"edge_triples"`` (list[tuple[int, int, int]]): Each entry is
              ``(i, j, rel)`` with ``i < j``.

    Raises:
        ValueError: If token values are outside valid ranges.
    """
    if tokens.shape != (vocab_config.seq_len,):
        raise ValueError(
            f"tokens shape must be ({vocab_config.seq_len},), "
            f"got {tokens.shape}"
        )
    if pad_mask.shape != (vocab_config.seq_len,):
        raise ValueError(
            f"pad_mask shape must be ({vocab_config.seq_len},), "
            f"got {pad_mask.shape}"
        )

    # --- Count real nodes ---
    num_rooms = int(pad_mask[: vocab_config.n_max].sum().item())

    # --- Extract node types ---
    node_types: list[int] = []
    for k in range(num_rooms):
        val = int(tokens[k].item())
        if not (0 <= val < len(NODE_TYPES)):
            raise ValueError(
                f"tokens[{k}] = {val} not in valid node range [0, {len(NODE_TYPES)})"
            )
        node_types.append(val)

    # --- Extract edge triples ---
    edge_triples: list[tuple[int, int, int]] = []
    for pos in range(vocab_config.n_edges):
        seq_idx = vocab_config.n_max + pos
        if not pad_mask[seq_idx]:
            # PAD position: skip
            continue
        val = int(tokens[seq_idx].item())
        if val == EDGE_NO_EDGE_IDX:
            # Real position but no adjacency relationship: skip
            continue
        # This is a real edge with a relationship
        i, j = vocab_config.edge_position_to_pair(pos)
        if not (0 <= val < len(EDGE_TYPES)):
            raise ValueError(
                f"edge at position {pos} (pair ({i}, {j})): "
                f"rel={val} not in [0, {len(EDGE_TYPES)})"
            )
        edge_triples.append((i, j, val))

    return {
        "num_rooms": num_rooms,
        "node_types": node_types,
        "edge_triples": edge_triples,
    }
