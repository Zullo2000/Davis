"""Transformer blocks with adaptive LayerNorm (adaLN-Zero) for time conditioning.

Implements the DiT-style (Peebles & Xie, ICCV 2023) transformer block where
time-dependent modulation parameters (shift, scale, gate) are predicted from
the conditioning signal and used to modulate LayerNorm outputs. Zero-initialization
of the modulation layer makes each block start as a standard transformer; the
model gradually learns to use time conditioning during training.

Uses PyTorch 2.0+ ``F.scaled_dot_product_attention`` (SDPA) rather than flash
attention, since the sequence length is only 36 tokens for RPLAN.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention using F.scaled_dot_product_attention.

    Uses a single combined projection for Q, K, V for efficiency, then
    reshapes for multi-head attention. PAD positions are masked via an
    additive float mask so that no token attends to PAD keys.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads. d_model must be divisible by n_heads.
        dropout: Attention dropout probability.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = dropout

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """Args:
            x: (B, S, d_model) input sequence.
            attn_mask: (B, S) boolean mask where True = IGNORE (PAD position).
                Converted to (B, 1, 1, S) float mask with -inf for True
                positions for use with F.scaled_dot_product_attention.

        Returns:
            (B, S, d_model) attended output.
        """
        B, S, D = x.shape

        # Combined QKV projection
        qkv = self.qkv_proj(x)  # (B, S, 3*d_model)
        qkv = qkv.reshape(B, S, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, S, head_dim)
        q, k, v = qkv.unbind(0)  # each: (B, n_heads, S, head_dim)

        # Convert boolean pad mask to additive attention mask for SDPA
        # (B,1,1,S) broadcasts across all heads and query positions,
        # ensuring no real token attends to PAD key positions.
        sdpa_mask = None
        if attn_mask is not None:
            sdpa_mask = torch.zeros(
                B, 1, 1, S, dtype=x.dtype, device=x.device
            )
            sdpa_mask.masked_fill_(attn_mask[:, None, None, :], float("-inf"))

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B, n_heads, S, head_dim)

        # Reshape back to (B, S, d_model)
        attn_output = attn_output.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(attn_output)


class AdaLNBlock(nn.Module):
    """Transformer block with adaptive LayerNorm zero-initialization.

    Each block modulates its LayerNorm outputs with 6 parameters predicted
    from the conditioning signal: (shift_msa, scale_msa, gate_msa, shift_mlp,
    scale_mlp, gate_mlp). The modulation layer is zero-initialized so the
    block starts as a standard transformer — at init, modulation is identity
    and gated residuals contribute nothing.

    This is the "adaLN-Zero" design from the DiT paper (Peebles & Xie,
    "Scalable Diffusion Models with Transformers", ICCV 2023).

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        cond_dim: Conditioning dimension (from timestep embedding).
        mlp_ratio: FFN hidden dim multiplier. Default 4.
        dropout: Dropout probability. Default 0.1.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        cond_dim: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Attention sub-block
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        # FFN sub-block
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model, bias=True),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)

        # adaLN modulation: 6 parameters from conditioning
        # Zero-init ensures block starts as identity (no modulation, zero gate)
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * d_model, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(
        self,
        x: Tensor,
        c: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """Args:
            x: (B, S, d_model) sequence.
            c: (B, d_model) conditioning vector (from TimestepEmbedding + SiLU).
            attn_mask: (B, S) boolean mask where True = IGNORE.

        Returns:
            (B, S, d_model) output sequence.
        """
        # Compute 6 modulation parameters from conditioning
        modulation = self.adaLN_modulation(c).unsqueeze(1)  # (B, 1, 6*d_model)
        (
            shift_msa, scale_msa, gate_msa,
            shift_mlp, scale_mlp, gate_mlp,
        ) = modulation.chunk(6, dim=2)  # each: (B, 1, d_model)

        # Attention sub-block with adaLN modulation
        # Formula: x_mod = LayerNorm(x) * (1 + scale) + shift
        x_norm = self.norm1(x)
        x_modulated = x_norm * (1 + scale_msa) + shift_msa
        x_attn = self.attn(x_modulated, attn_mask)
        x = x + gate_msa * self.dropout1(x_attn)

        # FFN sub-block with adaLN modulation
        x_norm2 = self.norm2(x)
        x_modulated2 = x_norm2 * (1 + scale_mlp) + shift_mlp
        x_ffn = self.mlp(x_modulated2)
        x = x + gate_mlp * self.dropout2(x_ffn)

        return x
