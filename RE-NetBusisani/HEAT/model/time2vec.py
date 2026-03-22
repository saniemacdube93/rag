"""
Time2Vec: Learning a Vector Representation of Time
Kazemi et al., 2019 — https://arxiv.org/abs/1907.05321

Encodes a scalar timestamp τ into a k-dimensional vector:
    Time2Vec(τ)[i] = ω_i · τ + φ_i               (i = 0, linear)
    Time2Vec(τ)[i] = sin(ω_i · τ + φ_i)           (i = 1..k, periodic)

Key insight: periodic components capture cyclical patterns (daily, weekly, seasonal),
while the linear component captures monotonic time progression.
"""

import torch
import torch.nn as nn
import math


class Time2Vec(nn.Module):
    """
    Learnable continuous-time encoding.

    Args:
        out_dim (int): dimension of the output time vector (k in the paper)
        activation (str): periodic activation — 'sin' or 'cos'
    """

    def __init__(self, out_dim: int, activation: str = 'sin'):
        super(Time2Vec, self).__init__()
        assert out_dim >= 2, "out_dim must be >= 2 (1 linear + at least 1 periodic)"

        self.out_dim = out_dim
        self.activation = torch.sin if activation == 'sin' else torch.cos

        # Linear (trend) component — scalar τ → scalar
        self.w0 = nn.Parameter(torch.randn(1) * 0.01)
        self.phi0 = nn.Parameter(torch.zeros(1))

        # Periodic components — scalar τ → (out_dim - 1) vector
        self.W = nn.Parameter(torch.randn(out_dim - 1) * 0.01)
        self.Phi = nn.Parameter(torch.zeros(out_dim - 1))

        self._init_weights()

    def _init_weights(self):
        """Initialise frequencies to capture multiple timescales."""
        with torch.no_grad():
            # Spread frequencies across log-scale (daily to multi-year)
            frequencies = torch.logspace(-2, 2, self.out_dim - 1)
            self.W.data = frequencies * (2 * math.pi)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tau: (...,) tensor of time gaps (scalar or batched)
        Returns:
            time_enc: (..., out_dim) time encoding
        """
        tau = tau.float()
        if tau.dim() == 0:
            tau = tau.unsqueeze(0)

        # Expand tau for broadcasting: (..., 1)
        tau_exp = tau.unsqueeze(-1)

        # Linear trend component: (..., 1)
        linear = self.w0 * tau_exp[..., 0:1] + self.phi0

        # Periodic components: (..., out_dim-1)
        periodic = self.activation(self.W * tau_exp + self.Phi)

        # Concatenate: (..., out_dim)
        return torch.cat([linear, periodic], dim=-1)

    def __repr__(self):
        return (f"Time2Vec(out_dim={self.out_dim}, "
                f"activation={'sin' if self.activation == torch.sin else 'cos'})")
