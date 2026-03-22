# HEAT: Hierarchical Evolution-Aware Transformer

**HEAT** is a novel temporal knowledge graph (TKG) reasoning model that improves over RE-Net through five key innovations — all trained end-to-end in a single stage.

> **Reference model:** Jin et al., *Recurrent Event Network* (EMNLP 2020)  
> **Proposed for:** PhD thesis on Temporal Knowledge Graph Reasoning

---

## Quick Start

```bash
# Full pipeline: train → evaluate → visualize
bash run_heat.sh ICEWS18
```

Or step by step:

```bash
# 1. Train
python3 train.py -d ICEWS18 --n-hidden 200 --max-epochs 30 --batch-size 512

# 2. Evaluate
python3 test.py -d ICEWS18 --uncertainty

# 3. Generate all scientific figures
python3 visualize.py -d ICEWS18
```

---

## Architecture — How HEAT Works

```
Query: (s, r, ?, t_query)
           │
           ▼
┌──────────────────────────────┐
│  1. Entity History Lookup    │  ← past facts for entity s
│  H_s = {(o_i, r_i, t_i)}   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  2. Event Embedding          │
│  h_i = W·[e_s ‖ e_o ‖      │
│           e_r ‖ Time2Vec(Δt)]│  ← Δt = t_query − t_i
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  3. Hierarchical Temporal    │  Local heads  → per-entity history
│     Self-Attention (HTSA)    │  Global heads → graph-level context
│     + Temporal Decay Bias    │  M[i,j] = −λ·|t_i − t_j|
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  4. Relation-Semantic Fusion │  g = σ(W·[e_struct ‖ e_semantic])
│     (Gated)                  │  e_r = g⊙e_struct + (1−g)⊙e_semantic
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  5. Prediction Head          │  logits = W·[entity_repr ‖ e_r]
│     + MC Dropout Uncertainty │  (T forward passes → variance)
└──────────────────────────────┘
           │
           ▼
    P(o | s, r, t) + confidence σ
```

---

## Key Innovations Over RE-Net

| Feature | RE-Net | HEAT |
|---------|--------|------|
| Training | Two-stage (pretrain → train) | ✅ Single end-to-end stage |
| History encoder | GRU (fixed window) | ✅ Transformer + temporal bias |
| Time modelling | Discrete steps | ✅ Time2Vec (continuous) |
| Relation embeddings | Random init | ✅ Gated semantic fusion |
| Uncertainty | None | ✅ Monte Carlo Dropout |
| Loss | Cross-entropy | ✅ CE + InfoNCE + temporal consistency |

---

## Mathematical Formulas

### Time2Vec

$$\text{T2V}(\tau)[i] = \begin{cases} \omega_0\tau + \phi_0 & i=0 \\ \sin(\omega_i\tau + \phi_i) & i \geq 1 \end{cases}$$

### Temporal Attention Bias

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M_{time}\right) V$$

$$M_{time}[i,j] = -\lambda \cdot |t_i - t_j|$$

### Relation-Semantic Gating

$$g = \sigma\!\bigl(W_g[e_{struct} \,\|\, e_{sem}]\bigr)$$
$$e_r = g \odot e_{struct} + (1-g) \odot e_{sem}$$

### Total Training Loss

$$\mathcal{L} = \mathcal{L}_{pred} + \alpha\,\mathcal{L}_{contrast} + \beta\,\mathcal{L}_{time}$$

$$\mathcal{L}_{contrast} = -\log\frac{\exp(z_a \cdot z_+ / \tau)}{\sum_k \exp(z_a \cdot z_k / \tau)}$$

---

## Datasets

Uses the same data as RE-Net — no extra preprocessing needed.

| Dataset | Entities | Relations | Train facts |
|---------|----------|-----------|------------|
| ICEWS18 | 23,033 | 256 | 373,018 |
| ICEWS14 | 12,498 | 260 | 323,895 |
| GDELT | 7,691 | 240 | 1,734,399 |
| WIKI | 12,554 | 24 | 539,286 |
| YAGO | 10,623 | 10 | 161,540 |

---

## Project Structure

```
HEAT/
├── model/
│   ├── heat_model.py          # Main model
│   ├── time2vec.py            # Time2Vec encoding
│   ├── temporal_attention.py  # HTSA module
│   └── relation_fusion.py     # Relation gating
├── losses/
│   └── losses.py              # Combined HEAT loss
├── data_loader.py             # History builder (no pickle needed)
├── train.py                   # Training script
├── test.py                    # Evaluation script
├── visualize.py               # 11 scientific plots
├── run_heat.sh                # Full pipeline runner
└── results/
    └── <DATASET>/
        ├── heat_best.pth       # Best model checkpoint
        ├── training_history.json
        ├── test_results.json
        └── figures/            # PNG + PDF plots (300 DPI)
```

---

## Evaluation Metrics

All metrics are **filtered** (ground-truth answers excluded from ranking):

- **MRR** — Mean Reciprocal Rank  
- **Hits@1** — % queries where true entity is rank 1  
- **Hits@3, Hits@10** — same at rank 3 and 10

---

## Scientific Visualisations (auto-generated)

| Plot | File |
|------|------|
| Training loss curve | `01_training_loss.pdf` |
| MRR progression | `02_mrr_progression.pdf` |
| Rank distribution | `03_rank_distribution.pdf` |
| Confusion matrix | `04_confusion_matrix.pdf` |
| ROC-style curve | `05_roc_curve.pdf` |
| Model comparison bar | `06_model_comparison.pdf` |
| MRR heatmap | `07_mrr_heatmap.pdf` |
| Calibration (uncertainty) | `08_calibration.pdf` |
| t-SNE embeddings | `09_tsne_embeddings.pdf` |
| Ablation study | `10_ablation_study.pdf` |
| Summary figure | `11_summary_comparison.pdf` |

---

## Citation

```bibtex
@inproceedings{macdube2025heat,
  title     = {HEAT: Hierarchical Evolution-Aware Transformer for
               Temporal Knowledge Graph Reasoning},
  author    = {Macdube, Busisani and ...},
  booktitle = {Proceedings of ...},
  year      = {2025}
}
```

---

*HEAT is implemented in Python 3.9 + PyTorch. No CUDA required — fully CPU compatible.*
