"""
Hierarchical Temporal Self-Attention (HTSA)

The core of HEAT's local event encoder. Replaces RE-Net's GRU with a Transformer
that applies explicit temporal bias to the attention scores.

Attention formula:
    A(Q, K, V) = softmax((QK^T / sqrt(d_k)) + M_time) · V

where M_time[i, j] = -λ · |τ_i - τ_j| (temporal decay bias)

Multi-head split:
  - Local heads: attend over per-entity history (replaces GRU)
  - Global heads: attend over graph-level aggregated features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TemporalMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with additive temporal bias.

    M_time[i, j] = -lambda * |t_i - t_j|

    This penalises attending to events far apart in time, creating a
    soft recency bias that the GRU cannot represent explicitly.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 temporal_lambda: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.temporal_lambda = nn.Parameter(torch.tensor(temporal_lambda))

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for lin in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(lin.weight)

    def _temporal_bias(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal bias matrix.
        Args:
            timestamps: (seq_len,) or (batch, seq_len) tensor of time values
        Returns:
            bias: (seq_len, seq_len) or (batch, seq_len, seq_len)
        """
        if timestamps.dim() == 1:
            t_i = timestamps.unsqueeze(1)   # (L, 1)
            t_j = timestamps.unsqueeze(0)   # (1, L)
        else:
            t_i = timestamps.unsqueeze(2)   # (B, L, 1)
            t_j = timestamps.unsqueeze(1)   # (B, 1, L)

        # Negative absolute time distance scaled by learned lambda
        bias = -torch.abs(self.temporal_lambda) * torch.abs(t_i - t_j)
        return bias

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor,
                mask: torch.Tensor = None):
        """
        Args:
            x:          (seq_len, d_model) or (batch, seq_len, d_model)
            timestamps: (seq_len,)         or (batch, seq_len)
            mask:       optional attention mask (True = masked out)
        Returns:
            out:        same shape as x
            attn_weights: (n_heads, seq_len, seq_len) for visualization
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)           # (1, L, d_model)
            timestamps = timestamps.unsqueeze(0)
            squeeze = True

        B, L, _ = x.shape
        residual = x

        # Linear projections → reshape to (B, n_heads, L, d_k)
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention scores: (B, n_heads, L, L)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Add temporal bias: (B, L, L) → broadcast over heads
        t_bias = self._temporal_bias(timestamps)      # (B, L, L)
        scores = scores + t_bias.unsqueeze(1)         # (B, n_heads, L, L)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Context: (B, n_heads, L, d_k) → (B, L, d_model)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.W_o(context)

        # Residual + layer norm
        out = self.layer_norm(out + residual)

        if squeeze:
            out = out.squeeze(0)
            attn = attn.squeeze(0)

        return out, attn


class HierarchicalTemporalAttention(nn.Module):
    """
    Two-layer HTSA with FFN — equivalent to a Transformer encoder block.

    Architecture:
        Layer 1: Local temporal attention (n_local_heads heads)
        Layer 2: Global temporal attention (n_global_heads heads)
        Each layer followed by FFN + LayerNorm + Dropout
    """

    def __init__(self, d_model: int, n_local_heads: int = 4,
                 n_global_heads: int = 4, ffn_dim: int = None,
                 dropout: float = 0.1, temporal_lambda: float = 0.1):
        super().__init__()

        ffn_dim = ffn_dim or 4 * d_model
        n_heads = n_local_heads + n_global_heads

        # Two-level attention
        self.local_attn = TemporalMultiHeadAttention(
            d_model, n_local_heads, dropout, temporal_lambda)
        self.global_attn = TemporalMultiHeadAttention(
            d_model, n_global_heads, dropout, temporal_lambda)

        # Feed-forward network after each attention layer
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

        # Gating between local and global outputs
        self.gate = nn.Linear(2 * d_model, d_model)

    def forward(self, local_x: torch.Tensor, global_x: torch.Tensor,
                timestamps: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            local_x:    (L, d_model) or (B, L, d_model)
            global_x:   (L, d_model) or (B, L, d_model)
            timestamps: (L,)         or (B, L)
            mask:       optional bool mask — True = padded position
        Returns:
            out:         (d_model,) or (B, d_model) — final fused representation
            local_attn:  attention weights for visualization
            global_attn: attention weights for visualization
        """
        # Local attention pass
        local_out, local_attn = self.local_attn(local_x, timestamps, mask)

        # Global attention pass (if global context provided)
        global_out, global_attn = self.global_attn(global_x, timestamps, mask)

        # Gated fusion of local and global
        gate_w = torch.sigmoid(self.gate(torch.cat([local_out, global_out], dim=-1)))
        fused = gate_w * local_out + (1 - gate_w) * global_out

        # FFN + residual
        out = self.ffn_norm(fused + self.ffn(fused))

        # Pool to single vector — handle both batched and unbatched
        if out.dim() == 3:  # batched: (B, L, h_dim) → (B, h_dim)
            if mask is not None:
                # Masked mean: only average over real (non-padded) positions
                valid = (~mask).float().unsqueeze(-1)          # (B, L, 1)
                entity_repr = (out * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            else:
                entity_repr = out.mean(dim=1)                  # (B, h_dim)
        else:               # unbatched: (L, h_dim) → (h_dim,)
            entity_repr = out.mean(dim=0)

        return entity_repr, local_attn, global_attn
