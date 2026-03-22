"""
HEAT: Hierarchical Evolution-Aware Transformer
Main model class — end-to-end joint training (no pretrain stage needed).

Architecture overview:
──────────────────────────────────────────────────────
  For a query (s, r, ?, t_q):

  1. Entity history H_s = {(o_i, r_i, t_i)} for subject s
  2. Per-event embedding:
       h_i = W · [e_s || e_o_i || e_{r_i} || Time2Vec(t_q - t_i)]
  3. Hierarchical Temporal Self-Attention (local + global heads)
       entity_repr = HTSA(h_1, ..., h_L; t_1, ..., t_L)
  4. Final prediction:
       logits = W_pred · [entity_repr || rel_embed]
       P(o|s,r,t) = softmax(logits)
──────────────────────────────────────────────────────

MC Dropout at test time gives calibrated uncertainty estimates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from .time2vec import Time2Vec
from .temporal_attention import HierarchicalTemporalAttention
from .relation_fusion import RelationSemanticFusion

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HEATModel(nn.Module):
    """
    Main HEAT model for temporal knowledge graph link prediction.

    Args:
        num_nodes (int):      number of entities |E|
        num_rels (int):       number of relation types |R|
        h_dim (int):          embedding dimensionality
        time_dim (int):       Time2Vec output dimension
        n_local_heads (int):  local attention heads
        n_global_heads (int): global attention heads
        seq_len (int):        max history length (τ)
        dropout (float):      dropout rate
        mc_samples (int):     Monte Carlo samples for uncertainty
    """

    def __init__(self, num_nodes: int, num_rels: int, h_dim: int = 200,
                 time_dim: int = 32, n_local_heads: int = 4,
                 n_global_heads: int = 4, seq_len: int = 10,
                 dropout: float = 0.3, mc_samples: int = 20):
        super(HEATModel, self).__init__()

        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.h_dim = h_dim
        self.time_dim = time_dim
        self.seq_len = seq_len
        self.mc_samples = mc_samples

        # ── Entity Embeddings ──────────────────────────────────────────────
        self.ent_embeds = nn.Embedding(num_nodes, h_dim)
        nn.init.xavier_uniform_(self.ent_embeds.weight)

        # ── Relation-Semantic Fusion ───────────────────────────────────────
        # Separate forward/inverse embeddings (2 * num_rels)
        self.rel_fusion_fwd = RelationSemanticFusion(num_rels, h_dim, dropout)
        self.rel_fusion_inv = RelationSemanticFusion(num_rels, h_dim, dropout)

        # ── Continuous Time Encoding ──────────────────────────────────────
        self.time_encoder = Time2Vec(time_dim)

        # ── Event Embedding Projection ────────────────────────────────────
        # Input: [e_s(h_dim) || e_o(h_dim) || e_r(h_dim) || time2vec(time_dim)]
        self.event_proj = nn.Sequential(
            nn.Linear(3 * h_dim + time_dim, h_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Hierarchical Temporal Self-Attention ──────────────────────────
        self.htsa = HierarchicalTemporalAttention(
            d_model=h_dim,
            n_local_heads=n_local_heads,
            n_global_heads=n_global_heads,
            ffn_dim=4 * h_dim,
            dropout=dropout,
        )

        # ── Empty History Embedding (when no history exists) ──────────────
        self.empty_hist_embed = nn.Parameter(torch.zeros(h_dim))

        # ── Prediction Head ───────────────────────────────────────────────
        # Input: [entity_repr(h_dim) || rel_embed(h_dim)] → num_nodes scores
        self.pred_head = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, num_nodes),
        )

        self.dropout = nn.Dropout(dropout)
        self.criterion = nn.CrossEntropyLoss()
        self.layer_norm = nn.LayerNorm(h_dim)

        # ── Store last attention weights for visualization ─────────────────
        self._last_attn_local = None
        self._last_attn_global = None

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: build event sequence tensor for a single entity
    # ─────────────────────────────────────────────────────────────────────────

    def _encode_history(self, s_id: int, history: list, history_t: list,
                        rel_embeds: torch.Tensor, t_query: float):
        """
        Encode per-entity history into a sequence of event embeddings.

        Args:
            s_id:      subject entity ID
            history:   list of numpy arrays [(o, r), ...] per timestep
            history_t: list of timestamps
            rel_embeds:(num_rels, h_dim)
            t_query:   query timestamp (float)

        Returns:
            event_seq: (L, h_dim)   — encoded events (L ≤ seq_len)
            time_seq:  (L,)         — time gaps for attention bias
        """
        if len(history) == 0:
            return None, None

        e_s = self.ent_embeds.weight[s_id]  # (h_dim,)

        event_vecs, time_gaps = [], []
        for facts, t in zip(history[-self.seq_len:], history_t[-self.seq_len:]):
            dt = max(float(t_query) - float(t), 0.0)
            time_gaps.append(dt)

            # facts is array of shape (num_facts, 2) — (r, o)
            if len(facts) == 0:
                continue
            facts_t = torch.LongTensor(facts).to(_device)
            r_ids = facts_t[:, 0]
            o_ids = facts_t[:, 1]

            e_o = self.ent_embeds(o_ids).mean(dim=0)       # (h_dim,)
            e_r = rel_embeds[r_ids].mean(dim=0)             # (h_dim,)
            dt_tensor = torch.tensor([dt], dtype=torch.float32).to(_device)
            t_enc = self.time_encoder(dt_tensor).squeeze(0) # (time_dim,)

            event_input = torch.cat([e_s, e_o, e_r, t_enc], dim=0)  # (3h+td,)
            event_vecs.append(self.event_proj(event_input))

        if len(event_vecs) == 0:
            return None, None

        event_seq = torch.stack(event_vecs, dim=0)          # (L, h_dim)
        time_seq = torch.tensor(time_gaps[:len(event_vecs)],
                                dtype=torch.float32).to(_device)
        return event_seq, time_seq

    def _encode_history_batch(self, src_ids: torch.Tensor, history_list: tuple,
                              rel_embeds: torch.Tensor, t_queries: torch.Tensor):
        """
        Batch-encode histories for all B samples using F.pad + torch.stack
        so gradients flow properly through event_proj / embeddings.

        Returns:
            event_batch: (B, max_L, h_dim) — padded event embeddings
            time_batch:  (B, max_L)        — padded time gaps (no grad needed)
            pad_mask:    (B, max_L) bool   — True where padding
            has_hist:    (B,) bool         — False where entity has no history
        """
        B = len(src_ids)
        encoded, times = [], []

        for i in range(B):
            ev, tm = self._encode_history(
                src_ids[i].item(),
                history_list[0][i],
                history_list[1][i],
                rel_embeds,
                t_queries[i].item(),
            )
            encoded.append(ev)
            times.append(tm)

        lengths = [ev.shape[0] if ev is not None else 0 for ev in encoded]
        max_L = max(lengths) if any(l > 0 for l in lengths) else 1

        padded_evs, padded_tms, pad_mask_rows, has_hist_flags = [], [], [], []

        for ev, tm, L_i in zip(encoded, times, lengths):
            if L_i > 0:
                pad_len = max_L - L_i
                # F.pad is differentiable — pads trailing dims first
                padded_ev = F.pad(ev, (0, 0, 0, pad_len))          # (max_L, h_dim)
                padded_tm = F.pad(tm, (0, pad_len))                 # (max_L,)
                mask_row  = torch.zeros(max_L, dtype=torch.bool, device=_device)
                mask_row[L_i:] = True                               # True = padding
                has_hist_flags.append(True)
            else:
                padded_ev = torch.zeros(max_L, self.h_dim, device=_device)
                padded_tm = torch.zeros(max_L, device=_device)
                mask_row  = torch.ones(max_L, dtype=torch.bool, device=_device)
                has_hist_flags.append(False)

            padded_evs.append(padded_ev)
            padded_tms.append(padded_tm)
            pad_mask_rows.append(mask_row)

        # Stack into batch tensors — fully differentiable!
        event_batch = torch.stack(padded_evs, dim=0)                # (B, max_L, h_dim)
        time_batch  = torch.stack(padded_tms, dim=0).detach()       # (B, max_L) no grad
        pad_mask    = torch.stack(pad_mask_rows, dim=0)             # (B, max_L) bool
        has_hist    = torch.tensor(has_hist_flags, dtype=torch.bool,
                                   device=_device)                  # (B,)

        return event_batch, time_batch, pad_mask, has_hist


    # ─────────────────────────────────────────────────────────────────────────
    # Forward: Training mode (batch of triplets)
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, triplets: torch.Tensor, s_history: tuple, o_history: tuple,
                subject: bool = True):
        """
        Vectorised training forward pass — processes the full batch in one
        HTSA call instead of sequentially per sample.

        Args:
            triplets:  (B, 4) tensor — [s, r, o, t]
            s_history: (hist_list, hist_t_list) for subject direction
            o_history: (hist_list, hist_t_list) for object direction
            subject:   if True predict object, else predict subject

        Returns:
            loss: scalar training loss
        """
        if subject:
            src = triplets[:, 0]
            r   = triplets[:, 1]
            tgt = triplets[:, 2]
            history_list = s_history
            rel_embeds = self.rel_fusion_fwd.get_all_embeddings()
        else:
            src = triplets[:, 2]
            r   = triplets[:, 1]
            tgt = triplets[:, 0]
            history_list = o_history
            rel_embeds = self.rel_fusion_inv.get_all_embeddings()

        t_queries = triplets[:, 3].float()

        # ── Batch-encode all histories → (B, max_L, h_dim) ────────────────
        event_batch, time_batch, pad_mask, has_hist = self._encode_history_batch(
            src, history_list, rel_embeds, t_queries)

        # ── Single HTSA call for the whole batch ───────────────────────────
        # Global context: mean of all entities' event sequences in the batch,
        # broadcast back to each sample so global heads see cross-entity signals.
        global_ctx = event_batch.mean(dim=0, keepdim=True).expand(B, -1, -1).detach()
        entity_reprs, self._last_attn_local, self._last_attn_global = \
            self.htsa(event_batch, global_ctx, time_batch, mask=pad_mask)
        # entity_reprs: (B, h_dim)

        # Replace entities with no history with the learned empty embedding
        empty = self.empty_hist_embed.unsqueeze(0).expand(len(src), -1)  # (B, h_dim)
        entity_reprs = torch.where(has_hist.unsqueeze(1), entity_reprs, empty)

        # ── Predict ────────────────────────────────────────────────────────
        rel_vecs = rel_embeds[r]                              # (B, h_dim)
        combined = torch.cat([entity_reprs, rel_vecs], dim=-1)  # (B, 2*h_dim)
        logits   = self.pred_head(combined)                   # (B, num_nodes)

        return logits, entity_reprs, tgt

    # ─────────────────────────────────────────────────────────────────────────
    # Predict: Test/eval mode — single sample, returns scores
    # ─────────────────────────────────────────────────────────────────────────

    def predict_scores(self, s_id: int, r_id: int, t_q: float,
                       history: list, history_t: list,
                       subject: bool = True) -> torch.Tensor:
        """
        Returns raw logits over all entities for ranking.

        Args:
            s_id:      subject (or object for inverse) entity ID
            r_id:      relation ID
            t_q:       query timestamp
            history:   entity history (list of fact arrays)
            history_t: entity history timestamps
            subject:   True = predict object, False = predict subject

        Returns:
            scores: (num_nodes,) logits
        """
        if subject:
            rel_embeds = self.rel_fusion_fwd.get_all_embeddings()
        else:
            rel_embeds = self.rel_fusion_inv.get_all_embeddings()

        event_seq, time_seq = self._encode_history(
            s_id, history, history_t, rel_embeds, t_q)

        if event_seq is None:
            entity_repr = self.empty_hist_embed
        else:
            entity_repr, self._last_attn_local, self._last_attn_global = \
                self.htsa(event_seq, event_seq, time_seq)

        r_vec = rel_embeds[r_id]
        combined = torch.cat([entity_repr, r_vec], dim=0).unsqueeze(0)
        logits = self.pred_head(combined).squeeze(0)
        return logits

    # ─────────────────────────────────────────────────────────────────────────
    # MC Dropout Uncertainty Estimation
    # ─────────────────────────────────────────────────────────────────────────

    def predict_with_uncertainty(self, s_id: int, r_id: int, t_q: float,
                                 history: list, history_t: list,
                                 subject: bool = True):
        """
        Monte Carlo Dropout inference for uncertainty quantification.

        Returns:
            mean_probs:  (num_nodes,) — mean probability across MC samples
            uncertainty: (num_nodes,) — variance across MC samples
        """
        self.train()  # Enable dropout for MC sampling
        all_probs = []

        with torch.no_grad():
            for _ in range(self.mc_samples):
                logits = self.predict_scores(
                    s_id, r_id, t_q, history, history_t, subject)
                probs = torch.softmax(logits, dim=-1)
                all_probs.append(probs.unsqueeze(0))

        self.eval()
        all_probs = torch.cat(all_probs, dim=0)         # (T, num_nodes)
        mean_probs = all_probs.mean(dim=0)              # (num_nodes,)
        uncertainty = all_probs.var(dim=0)              # (num_nodes,)
        return mean_probs, uncertainty
