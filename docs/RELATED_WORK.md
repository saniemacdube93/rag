# Related Work & Positioning

## Where TempoRAG Sits

```
                    Training Required?
                    YES              NO
               ┌─────────────┬──────────────┐
  Uses KG  YES │  CRONKGQA   │  TempoRAG    │ ← OUR POSITION
               │  TempoQR    │  (this work) │
               │  GenTKGQA   │              │
               │  TimeR4     │              │
               │  T-GRAG     │              │
               ├─────────────┼──────────────┤
  Uses KG  NO  │  Fine-tuned │  LLM-Direct  │
               │  LLMs       │  LLM-CoT     │
               └─────────────┴──────────────┘
```

TempoRAG occupies the **under-explored** quadrant: KG-grounded + training-free.

---

## Key Papers & How We Differ

| Paper | Year | Training | Temporal Scoring | Our Difference |
|-------|------|----------|-----------------|----------------|
| CRONKGQA | 2021 | Yes (KGE) | None | Training-free; explicit TRS |
| TempoQR | 2022 | Yes (KGE+T) | Implicit in embeddings | Training-free |
| GenTKGQA | 2024 | No training BUT needs GNN | 2-stage with constraints | Lighter; no GNN |
| TimeR4 | 2024 | Yes (contrastive) | Reranking via trained model | Training-free |
| RTQA | 2025 | No (but needs LLM decomp) | None | Adds TRS + TCV |
| T-GRAG | 2025 | Partial (community det.) | 5-component heavy system | Simpler 3-stage |
| **TempoRAG** | **2026** | **No** | **Explicit TRS (novel)** | — |
