"""
HEAT Combined Loss Function

Total loss:
    L_total = L_pred + α · L_contrast + β · L_time

1. L_pred  — cross-entropy link prediction loss
2. L_contrast — InfoNCE-style contrastive temporal loss
3. L_time  — temporal consistency regularisation (optional, β can be 0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HEATLoss(nn.Module):
    """
    Combined loss for HEAT training.

    Args:
        alpha (float): weight for contrastive loss
        beta (float):  weight for temporal consistency loss
        temperature (float): InfoNCE temperature τ
        n_negatives (int):   number of negative samples per anchor
    """

    def __init__(self, alpha: float = 0.1, beta: float = 0.05,
                 temperature: float = 0.07, n_negatives: int = 64):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.n_negatives = n_negatives
        self.ce = nn.CrossEntropyLoss()

    # ── 1. Prediction Loss ────────────────────────────────────────────────

    def prediction_loss(self, logits: torch.Tensor,
                        targets: torch.Tensor) -> torch.Tensor:
        """Standard cross-entropy over entity vocabulary."""
        return self.ce(logits, targets)

    # ── 2. Contrastive Temporal Loss (InfoNCE) ────────────────────────────

    def contrastive_loss(self, anchor_embeds: torch.Tensor,
                         pos_embeds: torch.Tensor,
                         all_embeds: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE contrastive loss.

        Entities co-occurring near each other in time should have similar
        representations; random entities should be dissimilar.

        L_contrast = -log [exp(z_a · z_+ / τ) / Σ_k exp(z_a · z_k / τ)]

        Args:
            anchor_embeds: (B, h_dim) — anchor entity embeddings
            pos_embeds:    (B, h_dim) — positive (co-occurring) embeddings
            all_embeds:    (N, h_dim) — full entity embedding matrix (negatives pool)

        Returns:
            contrastive loss scalar
        """
        if anchor_embeds.shape[0] == 0:
            return torch.tensor(0.0, device=anchor_embeds.device)

        B = anchor_embeds.shape[0]

        # L2 normalise
        z_a = F.normalize(anchor_embeds, dim=-1)    # (B, h_dim)
        z_p = F.normalize(pos_embeds, dim=-1)        # (B, h_dim)
        z_all = F.normalize(all_embeds, dim=-1)      # (N, h_dim)

        # Positive similarity: (B,)
        pos_sim = (z_a * z_p).sum(dim=-1) / self.temperature

        # Sample negatives (excluding anchor)
        n_neg = min(self.n_negatives, z_all.shape[0] - 1)
        neg_idx = torch.randperm(z_all.shape[0], device=z_all.device)[:n_neg]
        z_neg = z_all[neg_idx]                       # (n_neg, h_dim)

        # Negative similarities: (B, n_neg)
        neg_sim = torch.matmul(z_a, z_neg.t()) / self.temperature

        # InfoNCE: treat positive as class 0
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1+n_neg)
        labels = torch.zeros(B, dtype=torch.long, device=z_a.device)

        return F.cross_entropy(logits, labels)

    # ── 3. Temporal Consistency Loss ──────────────────────────────────────

    def temporal_consistency_loss(self, embeds_t: torch.Tensor,
                                  embeds_t_prev: torch.Tensor,
                                  neg_embeds: torch.Tensor) -> torch.Tensor:
        """
        Encourages entity embeddings to evolve smoothly over time:
            L_time = ||z_{s,t} - z_{s,t-1}||_2 - ||z_{s,t} - z_{s^-,t}||_2

        Near-entity pairs should be closer across time than random pairs.

        Args:
            embeds_t:      (B, h_dim) — entity embeds at time t
            embeds_t_prev: (B, h_dim) — same entity embeds at t-1
            neg_embeds:    (B, h_dim) — random entity embeds at t

        Returns:
            temporal consistency loss scalar
        """
        if embeds_t.shape[0] == 0:
            return torch.tensor(0.0, device=embeds_t.device)

        d_same = torch.norm(embeds_t - embeds_t_prev, p=2, dim=-1)  # (B,)
        d_diff = torch.norm(embeds_t - neg_embeds, p=2, dim=-1)     # (B,)

        # Hinge: encourage same-entity distance << different-entity distance
        margin = 1.0
        loss = F.relu(d_same - d_diff + margin).mean()
        return loss

    # ── Combined Forward ──────────────────────────────────────────────────

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                entity_embeds: torch.Tensor = None,
                pos_embeds: torch.Tensor = None,
                all_embeds: torch.Tensor = None,
                embeds_prev: torch.Tensor = None,
                neg_embeds: torch.Tensor = None) -> dict:
        """
        Compute total HEAT loss.

        Returns:
            dict with 'total', 'pred', 'contrast', 'time' losses
        """
        # Always compute prediction loss
        l_pred = self.prediction_loss(logits, targets)

        # Contrastive loss (only if embeddings provided)
        l_contrast = torch.tensor(0.0, device=logits.device)
        if entity_embeds is not None and pos_embeds is not None and all_embeds is not None:
            l_contrast = self.contrastive_loss(entity_embeds, pos_embeds, all_embeds)

        # Temporal consistency loss (optional)
        l_time = torch.tensor(0.0, device=logits.device)
        if embeds_prev is not None and neg_embeds is not None:
            l_time = self.temporal_consistency_loss(
                entity_embeds, embeds_prev, neg_embeds)

        total = l_pred + self.alpha * l_contrast + self.beta * l_time

        return {
            'total': total,
            'pred': l_pred.item(),
            'contrast': l_contrast.item(),
            'time': l_time.item(),
        }
