"""Soft violation utilities for guided bubble diagram generation.

Provides functions to build effective probability distributions from
partially-masked token sequences and model logits, enabling smooth
constraint violation scoring at any denoising step.

All probability tensors are float64 per spec Section 2.8.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from bd_gen.data.vocab import (
    EDGE_MASK_IDX,
    EDGE_NO_EDGE_IDX,
    EDGE_VOCAB_SIZE,
    NODE_MASK_IDX,
    NODE_VOCAB_SIZE,
    VocabConfig,
)


# ---------------------------------------------------------------------------
# build_effective_probs — single sample
# ---------------------------------------------------------------------------


def build_effective_probs(
    x_t: Tensor,
    node_logits: Tensor,
    edge_logits: Tensor,
    pad_mask: Tensor,
    vocab_config: VocabConfig,
) -> tuple[Tensor, Tensor]:
    """Build per-position probability distributions for a single sample.

    For each position:
      - PAD  → all zeros
      - Committed (not MASK) → one-hot on current token
      - MASK → softmax(logits)

    Args:
        x_t: ``(SEQ_LEN,)`` current tokens (may contain MASK).
        node_logits: ``(n_max, NODE_VOCAB_SIZE)`` raw logits.
        edge_logits: ``(n_edges, EDGE_VOCAB_SIZE)`` raw logits.
        pad_mask: ``(SEQ_LEN,)`` bool, True = real position.
        vocab_config: Vocabulary configuration.

    Returns:
        ``(node_probs, edge_probs)`` — float64 tensors matching the
        constraint interface shapes.
    """
    n_max = vocab_config.n_max

    # --- Nodes: (n_max, NODE_VOCAB_SIZE) ---
    node_tokens = x_t[:n_max]  # (n_max,)
    node_is_mask = node_tokens == NODE_MASK_IDX  # (n_max,)
    node_is_pad = ~pad_mask[:n_max]  # (n_max,)

    node_softmax = F.softmax(node_logits.double(), dim=-1)  # (n_max, V)
    node_onehot = F.one_hot(node_tokens, NODE_VOCAB_SIZE).double()  # (n_max, V)

    node_probs = torch.where(
        node_is_mask.unsqueeze(-1), node_softmax, node_onehot
    )
    node_probs[node_is_pad] = 0.0

    # --- Edges: (n_edges, EDGE_VOCAB_SIZE) ---
    edge_tokens = x_t[n_max:]  # (n_edges,)
    edge_is_mask = edge_tokens == EDGE_MASK_IDX  # (n_edges,)
    edge_is_pad = ~pad_mask[n_max:]  # (n_edges,)

    edge_softmax = F.softmax(edge_logits.double(), dim=-1)  # (n_edges, V)
    edge_onehot = F.one_hot(edge_tokens, EDGE_VOCAB_SIZE).double()  # (n_edges, V)

    edge_probs = torch.where(
        edge_is_mask.unsqueeze(-1), edge_softmax, edge_onehot
    )
    edge_probs[edge_is_pad] = 0.0

    return node_probs, edge_probs


# ---------------------------------------------------------------------------
# build_effective_probs_batch — K*B candidates
# ---------------------------------------------------------------------------


def build_effective_probs_batch(
    x_t: Tensor,
    node_logits: Tensor,
    edge_logits: Tensor,
    pad_mask: Tensor,
    vocab_config: VocabConfig,
) -> tuple[Tensor, Tensor]:
    """Batched version of :func:`build_effective_probs`.

    Args:
        x_t: ``(KB, SEQ_LEN)`` current tokens.
        node_logits: ``(KB, n_max, NODE_VOCAB_SIZE)`` raw logits.
        edge_logits: ``(KB, n_edges, EDGE_VOCAB_SIZE)`` raw logits.
        pad_mask: ``(KB, SEQ_LEN)`` bool, True = real position.
        vocab_config: Vocabulary configuration.

    Returns:
        ``(node_probs, edge_probs)`` — float64, shapes
        ``(KB, n_max, NODE_VOCAB_SIZE)`` and ``(KB, n_edges, EDGE_VOCAB_SIZE)``.
    """
    n_max = vocab_config.n_max

    # --- Nodes ---
    node_tokens = x_t[:, :n_max]  # (KB, n_max)
    node_is_mask = node_tokens == NODE_MASK_IDX  # (KB, n_max)
    node_is_pad = ~pad_mask[:, :n_max]  # (KB, n_max)

    node_softmax = F.softmax(node_logits.double(), dim=-1)
    node_onehot = F.one_hot(node_tokens, NODE_VOCAB_SIZE).double()

    node_probs = torch.where(
        node_is_mask.unsqueeze(-1), node_softmax, node_onehot
    )
    node_probs[node_is_pad] = 0.0

    # --- Edges ---
    edge_tokens = x_t[:, n_max:]  # (KB, n_edges)
    edge_is_mask = edge_tokens == EDGE_MASK_IDX
    edge_is_pad = ~pad_mask[:, n_max:]

    edge_softmax = F.softmax(edge_logits.double(), dim=-1)
    edge_onehot = F.one_hot(edge_tokens, EDGE_VOCAB_SIZE).double()

    edge_probs = torch.where(
        edge_is_mask.unsqueeze(-1), edge_softmax, edge_onehot
    )
    edge_probs[edge_is_pad] = 0.0

    return node_probs, edge_probs


# ---------------------------------------------------------------------------
# hard_decode_x0 — argmax decode for hard reward mode
# ---------------------------------------------------------------------------


def hard_decode_x0(
    x_t: Tensor,
    node_logits: Tensor,
    edge_logits: Tensor,
    pad_mask: Tensor,
    vocab_config: VocabConfig,
) -> Tensor:
    """Hard-decode x̂_0: committed positions keep token, MASK → argmax.

    Supports both single sample ``(SEQ_LEN,)`` and batched ``(KB, SEQ_LEN)``
    inputs.  PAD positions are left unchanged.

    Returns:
        Tensor of same shape as *x_t* with valid token indices at all
        real positions (no MASK tokens).
    """
    n_max = vocab_config.n_max
    result = x_t.clone()

    if x_t.dim() == 2:
        # Batched: (KB, SEQ_LEN)
        node_tokens = result[:, :n_max]
        edge_tokens = result[:, n_max:]
        node_mask_pos = node_tokens == NODE_MASK_IDX
        edge_mask_pos = edge_tokens == EDGE_MASK_IDX
        node_tokens[node_mask_pos] = node_logits.argmax(dim=-1)[node_mask_pos]
        edge_tokens[edge_mask_pos] = edge_logits.argmax(dim=-1)[edge_mask_pos]
    else:
        # Single: (SEQ_LEN,)
        node_tokens = result[:n_max]
        edge_tokens = result[n_max:]
        node_mask_pos = node_tokens == NODE_MASK_IDX
        edge_mask_pos = edge_tokens == EDGE_MASK_IDX
        node_tokens[node_mask_pos] = node_logits.argmax(dim=-1)[node_mask_pos]
        edge_tokens[edge_mask_pos] = edge_logits.argmax(dim=-1)[edge_mask_pos]

    return result


# ---------------------------------------------------------------------------
# _compute_adj_terms — shared helper for RequireAdj / ForbidAdj
# ---------------------------------------------------------------------------


def _compute_adj_terms(
    node_probs: Tensor,
    edge_probs: Tensor,
    pad_mask: Tensor,
    vocab_config: VocabConfig,
    type_a_idx: int,
    type_b_idx: int,
) -> Tensor:
    """Compute ``p_ij = p_types_ij * P_adj_ij`` for all edge positions.

    For each edge position ``(i, j)``:
      - ``p_types_ij``: joint probability that nodes i, j have the required
        types (both orderings if ``type_a != type_b``).
      - ``P_adj_ij``: probability of any spatial relationship (sum of the 10
        spatial type probabilities, i.e. ``1 - P(NO_EDGE)``).

    PAD edge positions are zeroed out.

    Args:
        node_probs: ``(n_max, NODE_VOCAB_SIZE)`` float64.
        edge_probs: ``(n_edges, EDGE_VOCAB_SIZE)`` float64.
        pad_mask: ``(seq_len,)`` bool.
        vocab_config: Vocabulary configuration.
        type_a_idx: Room type index for the first type.
        type_b_idx: Room type index for the second type.

    Returns:
        ``(n_edges,)`` float64 tensor of per-edge-position probabilities.
    """
    n_max = vocab_config.n_max
    n_edges = vocab_config.n_edges

    # Vectorized edge-pair indices (upper-triangle, row-major order)
    i_indices, j_indices = torch.triu_indices(n_max, n_max, offset=1)

    # P(node i is type_a) and P(node j is type_b)
    q_i_a = node_probs[i_indices, type_a_idx]  # (n_edges,)
    q_j_b = node_probs[j_indices, type_b_idx]  # (n_edges,)

    if type_a_idx == type_b_idx:
        p_types = q_i_a * q_j_b
    else:
        q_i_b = node_probs[i_indices, type_b_idx]
        q_j_a = node_probs[j_indices, type_a_idx]
        p_types = q_i_a * q_j_b + q_i_b * q_j_a

    # P(any spatial relationship) = sum of spatial types (indices 0..9)
    p_adj = edge_probs[:, :EDGE_NO_EDGE_IDX].sum(dim=-1)  # (n_edges,)

    p_ij = p_types * p_adj  # (n_edges,)

    # Zero out PAD edge positions
    edge_pad = ~pad_mask[n_max:]  # (n_edges,)
    p_ij = p_ij.clone()
    p_ij[edge_pad] = 0.0

    return p_ij
