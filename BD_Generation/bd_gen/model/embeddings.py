"""Embedding modules for the BD-Gen transformer denoiser.

Provides token embeddings (separate for nodes and edges since they have
different vocabularies), composite positional encoding that captures both
entity type (node vs edge) and structural position, and sinusoidal timestep
embedding with MLP projection.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from bd_gen.data.vocab import (
    EDGE_VOCAB_SIZE,
    NODE_VOCAB_SIZE,
    VocabConfig,
)


class NodeEmbedding(nn.Module):
    """Embed node token indices into d_model-dimensional vectors.

    Maps NODE_VOCAB_SIZE (15) discrete room-type indices (including MASK
    and PAD) into continuous vectors.

    Args:
        d_model: Embedding dimension.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(NODE_VOCAB_SIZE, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: (B, n_nodes) long tensor of node token indices.

        Returns:
            (B, n_nodes, d_model) float tensor.
        """
        return self.embedding(x)


class EdgeEmbedding(nn.Module):
    """Embed edge token indices into d_model-dimensional vectors.

    Maps EDGE_VOCAB_SIZE (13) discrete edge-type indices (including
    no-edge, MASK, and PAD) into continuous vectors.

    Args:
        d_model: Embedding dimension.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(EDGE_VOCAB_SIZE, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: (B, n_edges) long tensor of edge token indices.

        Returns:
            (B, n_edges, d_model) float tensor.
        """
        return self.embedding(x)


class CompositePositionalEncoding(nn.Module):
    """Learned positional encoding that captures entity type and graph structure.

    Our token sequence is a flattened graph (nodes then edges), not a natural
    ordered sequence. Standard positional indices would be meaningless.
    Instead, we encode structural information:

    For node position k (0 <= k < n_max):
        pos[k] = entity_type_emb(0) + node_index_emb(k)

    For edge position at (i, j) where 0 <= i < j < n_max:
        pos[n_max + flat_idx] = entity_type_emb(1)
                                + pair_index_emb(i) + pair_index_emb(j)

    The pair indices (i, j) for each edge are derived from
    VocabConfig.edge_position_to_pair() and precomputed once at construction
    time. Index tensors are registered as buffers for transparent device
    movement.

    Args:
        vocab_config: VocabConfig providing n_max, n_edges, seq_len, and
            edge_position_to_pair().
        d_model: Embedding dimension.
    """

    def __init__(self, vocab_config: VocabConfig, d_model: int) -> None:
        super().__init__()
        self.vocab_config = vocab_config
        n_max = vocab_config.n_max
        n_edges = vocab_config.n_edges

        # Learned embeddings
        self.entity_type_emb = nn.Embedding(2, d_model)  # 0=node, 1=edge
        self.node_index_emb = nn.Embedding(n_max, d_model)  # which room slot
        self.pair_index_emb = nn.Embedding(n_max, d_model)  # shared for endpoints

        # Precompute index tensors for efficient forward pass
        node_indices = torch.arange(n_max, dtype=torch.long)
        self.register_buffer("node_indices", node_indices)

        # Entity type indices: [0]*n_max + [1]*n_edges
        entity_types = torch.cat([
            torch.zeros(n_max, dtype=torch.long),
            torch.ones(n_edges, dtype=torch.long),
        ])
        self.register_buffer("entity_types", entity_types)

        # Edge endpoint indices: precompute (i, j) for each edge position
        edge_i = torch.zeros(n_edges, dtype=torch.long)
        edge_j = torch.zeros(n_edges, dtype=torch.long)
        for pos in range(n_edges):
            i, j = vocab_config.edge_position_to_pair(pos)
            edge_i[pos] = i
            edge_j[pos] = j
        self.register_buffer("edge_i", edge_i)
        self.register_buffer("edge_j", edge_j)

    def forward(self) -> Tensor:
        """Compute positional encoding for the full sequence.

        Returns:
            (seq_len, d_model) float tensor, broadcastable to
            (B, seq_len, d_model) when added to token embeddings.
        """
        n_max = self.vocab_config.n_max

        # Entity type component for entire sequence
        entity_enc = self.entity_type_emb(self.entity_types)  # (seq_len, d_model)

        # Node positional component
        node_pos_enc = torch.zeros_like(entity_enc)
        node_pos_enc[:n_max] = self.node_index_emb(self.node_indices)

        # Edge positional component: pair_index_emb(i) + pair_index_emb(j)
        edge_pos_enc = torch.zeros_like(entity_enc)
        edge_pos_enc[n_max:] = (
            self.pair_index_emb(self.edge_i) + self.pair_index_emb(self.edge_j)
        )

        return entity_enc + node_pos_enc + edge_pos_enc


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep encoding followed by MLP projection.

    Step 1: Sinusoidal positional encoding of continuous t in [0, 1],
    using multi-frequency basis from Vaswani et al. / GLIDE / DiT.

    Step 2: Two-layer MLP with SiLU activation projects into model space.

    Note: In BDDenoiser.forward(), an additional SiLU is applied AFTER
    this module: ``c = F.silu(timestep_embedding(t))``. This matches the
    DiT/DiDAPS convention where silu shapes the conditioning signal before
    it enters adaLN modulation blocks.

    Args:
        d_model: Output dimension (and MLP hidden dimension).
        frequency_embedding_size: Dimension of the sinusoidal encoding.
            Default 256.
    """

    def __init__(
        self,
        d_model: int,
        frequency_embedding_size: int = 256,
    ) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )

    @staticmethod
    def sinusoidal_encoding(t: Tensor, dim: int) -> Tensor:
        """Create sinusoidal timestep embeddings.

        Args:
            t: (B,) float32 tensor in [0, 1].
            dim: Output dimension (frequency_embedding_size).

        Returns:
            (B, dim) float32 tensor.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None, :]  # (B, half)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        """Args:
            t: (B,) float32 tensor of timesteps in [0, 1].

        Returns:
            (B, d_model) float tensor.
        """
        t_freq = self.sinusoidal_encoding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)
