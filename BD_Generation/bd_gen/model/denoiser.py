"""BDDenoiser: top-level denoiser model for bubble diagram generation.

Wraps separate node/edge embeddings, composite positional encoding,
timestep embedding, transformer blocks with adaLN-Zero, and dual
classification heads into a single nn.Module. The forward pass accepts
a flat token sequence, PAD mask, and timestep, producing per-position
logits over the node and edge vocabularies.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bd_gen.data.vocab import (
    EDGE_MASK_IDX,
    EDGE_PAD_IDX,
    EDGE_VOCAB_SIZE,
    NODE_MASK_IDX,
    NODE_PAD_IDX,
    NODE_VOCAB_SIZE,
    VocabConfig,
)
from bd_gen.model.embeddings import (
    CompositePositionalEncoding,
    EdgeEmbedding,
    NodeEmbedding,
    TimestepEmbedding,
)
from bd_gen.model.transformer import AdaLNBlock


class BDDenoiser(nn.Module):
    """MDLM transformer denoiser for bubble diagram generation.

    Accepts a sequence of token indices (nodes + edges), a boolean PAD mask,
    and a diffusion timestep. Produces per-position logits over the
    appropriate vocabulary for each position type.

    The model uses separate embedding tables for nodes and edges (since they
    have different vocabularies), a shared composite positional encoding,
    and adaLN-Zero transformer blocks for time conditioning.

    Args:
        d_model: Hidden dimension throughout the transformer.
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads per block.
        vocab_config: VocabConfig providing sequence layout parameters.
        cond_dim: Conditioning dimension. Defaults to d_model if None.
        mlp_ratio: FFN expansion ratio. Default 4.
        dropout: Dropout rate. Default 0.1.
        frequency_embedding_size: Sinusoidal embedding dim. Default 256.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        vocab_config: VocabConfig,
        cond_dim: int | None = None,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        frequency_embedding_size: int = 256,
    ) -> None:
        super().__init__()
        if cond_dim is None:
            cond_dim = d_model
        self.d_model = d_model
        self.vocab_config = vocab_config

        # Token embeddings (separate vocabularies for nodes and edges)
        self.node_embedding = NodeEmbedding(d_model)
        self.edge_embedding = EdgeEmbedding(d_model)

        # Positional encoding (captures entity type + graph structure)
        self.positional_encoding = CompositePositionalEncoding(vocab_config, d_model)

        # Timestep embedding (sinusoidal + MLP)
        self.timestep_embedding = TimestepEmbedding(d_model, frequency_embedding_size)

        # Transformer blocks with adaLN-Zero
        self.blocks = nn.ModuleList([
            AdaLNBlock(d_model, n_heads, cond_dim, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])

        # Final adaLN modulation (shift + scale only, no gate — no residual here)
        # Zero-init so initial output passes through LayerNorm unchanged
        self.final_norm = nn.LayerNorm(d_model)
        self.final_adaLN = nn.Linear(cond_dim, 2 * d_model, bias=True)
        self.final_adaLN.weight.data.zero_()
        self.final_adaLN.bias.data.zero_()

        # Classification heads — zero-init for uniform initial logits
        self.node_head = nn.Linear(d_model, NODE_VOCAB_SIZE)
        self.node_head.weight.data.zero_()
        self.node_head.bias.data.zero_()

        self.edge_head = nn.Linear(d_model, EDGE_VOCAB_SIZE)
        self.edge_head.weight.data.zero_()
        self.edge_head.bias.data.zero_()

    def _process_t(
        self,
        t: Tensor | float | int,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        """Normalize timestep to a 1D float32 tensor of shape (batch_size,).

        Accepts:
            - Python float or int: broadcast to all samples.
            - 0D tensor: broadcast to all samples.
            - 1D tensor of size 1: broadcast to all samples.
            - 1D tensor of size batch_size: used as-is.

        Returns:
            Tensor of shape (batch_size,), dtype=torch.float32, on device.

        Raises:
            ValueError: For invalid tensor shapes.
        """
        if isinstance(t, (float, int)):
            return torch.full(
                (batch_size,), float(t), dtype=torch.float32, device=device
            )

        t = torch.as_tensor(t, dtype=torch.float32, device=device)

        if t.dim() == 0:
            return t.expand(batch_size)

        if t.dim() == 1:
            if t.size(0) == 1:
                return t.expand(batch_size)
            if t.size(0) == batch_size:
                return t
            raise ValueError(
                f"1D timestep tensor has size {t.size(0)}, "
                f"expected 1 or batch_size={batch_size}"
            )

        raise ValueError(
            f"timestep must be scalar, 0D, or 1D tensor, got {t.dim()}D"
        )

    def forward(
        self,
        tokens: Tensor,
        pad_mask: Tensor,
        t: Tensor | float | int,
        condition: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass: tokens + timestep -> per-position logits.

        Args:
            tokens: (B, seq_len) long tensor of token indices.
                Positions [0:n_max] index into NODE_VOCAB (0-14).
                Positions [n_max:seq_len] index into EDGE_VOCAB (0-12).
            pad_mask: (B, seq_len) bool tensor. True = real position,
                False = PAD position.
            t: Timestep. Flexible input: float, int, 0D, 1D size 1,
                or 1D size B. Float in [0, 1].
            condition: Unused in v1. Placeholder for v2 cross-attention
                with house boundary features.

        Returns:
            Tuple of:
                - node_logits: (B, n_max, NODE_VOCAB_SIZE) float32.
                - edge_logits: (B, n_edges, EDGE_VOCAB_SIZE) float32.
        """
        B = tokens.size(0)
        n_max = self.vocab_config.n_max
        device = tokens.device

        # 1. Split tokens into node and edge parts
        node_tokens = tokens[:, :n_max]  # (B, n_max)
        edge_tokens = tokens[:, n_max:]  # (B, n_edges)

        # 2. Embed separately (different vocabularies)
        node_emb = self.node_embedding(node_tokens)  # (B, n_max, d_model)
        edge_emb = self.edge_embedding(edge_tokens)  # (B, n_edges, d_model)

        # 3. Concatenate into full sequence
        x = torch.cat([node_emb, edge_emb], dim=1)  # (B, seq_len, d_model)

        # 4. Add positional encoding (broadcasts from (seq_len, d_model))
        x = x + self.positional_encoding()

        # 5. Process timestep: sinusoidal + MLP + outer SiLU
        t_processed = self._process_t(t, B, device)
        c = F.silu(self.timestep_embedding(t_processed))  # (B, d_model)

        # 6. Create attention mask: True = IGNORE (inverted pad_mask)
        attn_mask = ~pad_mask  # (B, seq_len)

        # 7. Pass through transformer blocks
        for block in self.blocks:
            x = block(x, c, attn_mask)

        # 8. Final adaLN modulation + LayerNorm
        final_shift, final_scale = (
            self.final_adaLN(c).unsqueeze(1).chunk(2, dim=2)
        )  # each: (B, 1, d_model)
        x = self.final_norm(x) * (1 + final_scale) + final_shift

        # 9. Split back into node and edge features
        node_features = x[:, :n_max]  # (B, n_max, d_model)
        edge_features = x[:, n_max:]  # (B, n_edges, d_model)

        # 10. Classification heads
        node_logits = self.node_head(node_features)  # (B, n_max, NODE_VOCAB_SIZE)
        edge_logits = self.edge_head(edge_features)  # (B, n_edges, EDGE_VOCAB_SIZE)

        # 11. SUBS zero masking probabilities: prevent predicting MASK or PAD
        node_logits[:, :, NODE_MASK_IDX] = float('-inf')
        node_logits[:, :, NODE_PAD_IDX] = float('-inf')
        edge_logits[:, :, EDGE_MASK_IDX] = float('-inf')
        edge_logits[:, :, EDGE_PAD_IDX] = float('-inf')

        return node_logits, edge_logits
