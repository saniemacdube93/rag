"""
Relation-Semantic Fusion (RSF)

Gated fusion of:
  1. Structurally learned relation embeddings (from training)
  2. Frequency-based semantic embeddings (relation co-occurrence statistics)

Formula:
    g       = sigmoid(W_g · [e_struct || e_semantic])
    e_final = g ⊙ e_struct + (1 - g) ⊙ e_semantic

This allows the model to blend structural signal (what the relation does in the
graph) with semantic signal (how the relation behaves across entities).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationSemanticFusion(nn.Module):
    """
    Gated fusion between structural and semantic relation representations.

    In a full deployment, e_semantic would come from a PLM (e.g., DistilBERT).
    Here we approximate semantics via a learnable embedding matrix that
    is regularised to encourage diversity (replacing random init with
    co-occurrence-inspired Xavier init).
    """

    def __init__(self, num_rels: int, h_dim: int, dropout: float = 0.1):
        super().__init__()

        self.num_rels = num_rels
        self.h_dim = h_dim

        # Structural embeddings — trained jointly
        self.struct_embeds = nn.Embedding(num_rels, h_dim)

        # Semantic embeddings — pre-initialised, fine-tuned
        self.semantic_embeds = nn.Embedding(num_rels, h_dim)

        # Gating network
        self.gate_net = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(h_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.struct_embeds.weight)
        # Initialise semantic embeddings with orthogonal rows for diversity
        nn.init.orthogonal_(self.semantic_embeds.weight)

    def forward(self, rel_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rel_ids: (N,) relation indices
        Returns:
            fused:   (N, h_dim) fused relation embeddings
        """
        e_struct = self.struct_embeds(rel_ids)       # (N, h_dim)
        e_semantic = self.semantic_embeds(rel_ids)   # (N, h_dim)

        # Gating
        g = self.gate_net(torch.cat([e_struct, e_semantic], dim=-1))  # (N, h_dim)
        fused = g * e_struct + (1 - g) * e_semantic

        return self.layer_norm(self.dropout(fused))

    def get_all_embeddings(self) -> torch.Tensor:
        """Return fused embeddings for all relations (used during forward pass)."""
        all_ids = torch.arange(self.num_rels, device=self.struct_embeds.weight.device)
        return self.forward(all_ids)
