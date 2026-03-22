# TempoRAG: A Lightweight Training-Free Temporal Subgraph RAG Framework for KGQA

> **Target Venue**: EMNLP 2025 / ACL 2026 (findings track)
> **Hardware**: Apple M3 Max, 36GB Unified RAM — no GPU cluster required
> **Status**: Design phase

---

## 1. PROBLEM STATEMENT

Temporal Knowledge Graph Question Answering (TKGQA) requires reasoning over
facts that are valid only within specific time intervals. For example:

> "Who was the president of France before Macron?"
> "What was Apple's CEO during the 2008 financial crisis?"

**The core challenge**: existing RAG systems retrieve facts without temporal
awareness — they may surface facts that were true at *some* point but not
during the time window implied by the question.

**The training bottleneck**: state-of-the-art systems (TimeR4, T-GRAG, MemoTime)
require expensive training (contrastive learning, GNN training, fine-tuning).
This makes them inaccessible for researchers without GPU clusters.

**Our answer**: TempoRAG — a *training-free*, *lightweight* temporal subgraph
RAG framework that achieves competitive performance using only a frozen
quantized LLM and a novel temporal relevance scoring function.

---

## 2. PAPER TITLE (WORKING)

**"TempoRAG: Training-Free Temporal Subgraph Retrieval-Augmented Generation
for Knowledge Graph Question Answering"**

---

## 3. SYSTEM ARCHITECTURE (3-Stage Pipeline)

```
QUESTION
   │
   ▼
┌─────────────────────────────────────────┐
│  STAGE 1: Temporal Question Analysis    │
│  - Entity mention detection (spaCy)     │
│  - Temporal anchor extraction           │
│    (years, dates, durations, ordinals)  │
│  - Temporal operator classification     │
│    (BEFORE / AFTER / DURING / FIRST /  │
│     LAST / EQUAL / DURATION)            │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  STAGE 2: Time-Filtered Subgraph        │
│           Retrieval  [NOVEL]            │
│                                         │
│  2a. Entity Linking → KG anchor nodes  │
│  2b. k-hop Subgraph Extraction          │
│      (k=2 for simple, k=3 for complex)  │
│  2c. Temporal Relevance Scoring         │
│      TRS(triple) =                      │
│        α · time_proximity_score         │
│        + β · operator_match_score       │
│        + γ · entity_centrality_score    │
│  2d. Top-N triple selection             │
│      (N=20 default, adaptive by type)   │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  STAGE 3: Temporally-Grounded LLM Gen  │
│                                         │
│  3a. Chronological ordering of triples │
│  3b. Structured prompt construction    │
│  3c. Frozen LLM generation             │
│      (Llama-3.1-8B-Q4 via Ollama)      │
│  3d. Temporal Constraint Verification  │
│      (rule-based post-hoc check)  [NEW]│
└──────────────────┬──────────────────────┘
                   │
                   ▼
                ANSWER
```

### 3.1 Stage 1 — Temporal Question Analysis

**Input**: Raw natural language question
**Output**: (entities[], time_anchor, temporal_operator, question_type)

Components:
- **Entity Detection**: spaCy NER + entity linking to KG via string matching / BM25
- **Temporal Anchor Extraction**: regex + SUTime for dates/years/durations
- **Temporal Operator Classification**: lightweight classifier (6 classes) using
  keyword rules + small encoder (BERT-base, frozen, zero-shot)
  - BEFORE, AFTER, DURING, FIRST, LAST, EQUAL

### 3.2 Stage 2 — Time-Filtered Subgraph Retrieval [PRIMARY CONTRIBUTION]

**Input**: (entities[], time_anchor, temporal_operator)
**Output**: ranked list of temporal triples (subject, predicate, object, t_start, t_end)

**Temporal Relevance Score (TRS)** — the novel scoring function:

```
TRS(t) = α · Prox(t, t_anchor)
        + β · OpMatch(t, op)
        + γ · EntRel(t, q_entities)

where:
  Prox(t, t_anchor) = exp(-|midpoint(t) - t_anchor| / σ)
      → Gaussian decay from question time anchor

  OpMatch(t, op) = {1.0 if triple satisfies operator constraint,
                    0.5 if partially satisfies,
                    0.0 otherwise}
      → Hard/soft constraint matching for BEFORE/AFTER/DURING etc.

  EntRel(t, q_entities) = max(sim(t.subj, e), sim(t.obj, e)) for e in q_entities
      → Entity relevance via KG embedding similarity (TComplEx, pre-trained)

  α=0.4, β=0.35, γ=0.25  (tuned on validation set)
```

This scoring is **training-free** — we use a *pre-trained* TComplEx model
(already available on HuggingFace) only for EntityRel similarity, not for QA.

### 3.3 Stage 3 — Temporally-Grounded Generation

**Input**: top-N triples (chronologically sorted)
**Output**: final answer string

Prompt template:
```
[TEMPORAL CONTEXT - ordered by time]
- (Obama, presidentOf, USA, 2009-2017)
- (Trump, presidentOf, USA, 2017-2021)
- (Biden, presidentOf, USA, 2021-2025)

[QUESTION] Who was the US president during the 2008 financial crisis?
[TEMPORAL CONSTRAINT] Time anchor: 2008 | Operator: DURING
[ANSWER]
```

**Temporal Constraint Verifier (TCV)** — post-hoc rule-based check:
- Validates that predicted answer entity has a KG triple valid at t_anchor
- If validation fails → fall back to next candidate
- Simple lookup, no training needed

---

## 4. DATASETS

### Primary Evaluation Datasets

| Dataset | Size | KG Source | Question Types | Use in Paper |
|---------|------|-----------|----------------|--------------|
| **CronQuestions** | 410K total → use **10K subset** | Wikidata | Simple Entity, Simple Time, Before/After, First/Last, Time Join | Main benchmark |
| **TimeQuestions** | 16,859 (full) | Wikidata | Explicit/Implicit temporal, Ordinal | Full evaluation |
| **MultiTQ** | 500K → use **5K subset** | ICEWS05-15 | Simple/Complex, multi-granularity | Generalization test |

### Why These Three?
- **CronQuestions**: largest, most cited, all baselines reported here
- **TimeQuestions**: different distribution, tests generalization
- **MultiTQ**: different KG (event-based ICEWS vs. entity-based Wikidata),
  tests cross-domain robustness

### Data Access
```
CronQuestions : https://github.com/apoorvumang/CronKGQA
TimeQuestions : HuggingFace datasets
MultiTQ       : https://github.com/czy1999/MultiTQ
```

### Splits Used
```
CronQuestions : 8K train (for ablations only) / 1K val / 1K test
TimeQuestions : standard split (no training used)
MultiTQ       : 4K / 500 / 500
```

Note: "train" split used only for hyperparameter tuning (α,β,γ), not model training.

---

## 5. BASELINES

### Group A — Direct LLM (No KG)

| Baseline | Model | Description |
|----------|-------|-------------|
| **LLM-Direct** | Llama-3.1-8B-Q4 | Zero-shot, no retrieval |
| **LLM-CoT** | Llama-3.1-8B-Q4 | Chain-of-thought prompting, no KG |

Purpose: Show that KG grounding is necessary for temporal QA

### Group B — Static RAG (No Temporal Awareness)

| Baseline | Description |
|----------|-------------|
| **BM25-RAG** | BM25 retrieval of KG triples (verbalized) + LLM |
| **Dense-RAG** | FAISS + all-MiniLM-L6 embeddings + LLM |

Purpose: Show that *temporal-aware* retrieval is better than static retrieval

### Group C — Classic Temporal KGQA

| Baseline | Paper | Notes |
|----------|-------|-------|
| **CRONKGQA** | ACL 2021 | Reproduced using official code |
| **TempoQR** | AAAI 2022 | Reproduced using official code |

Purpose: Compare against embedding-based SOTA from early work

### Group D — Modern LLM-based TKGQA

| Baseline | Paper | Notes |
|----------|-------|-------|
| **GenTKGQA** | ACL 2024 | Reproduced / results from paper |
| **RTQA** | EMNLP 2025 | Reproduced using official code |

Purpose: Compare against current SOTA LLM-based methods

### Ablations (Our System Variants)

| Variant | What's Removed |
|---------|----------------|
| **TempoRAG-NoTRS** | Replace TRS with random triple selection |
| **TempoRAG-NoProx** | Remove Prox term from TRS (α=0) |
| **TempoRAG-NoOp** | Remove OpMatch term (β=0) |
| **TempoRAG-NoTCV** | Remove post-hoc verifier |
| **TempoRAG-k1** | k=1 hop subgraph only |
| **TempoRAG-full** | Full system (reported as main result) |

---

## 6. EVALUATION METRICS

### Standard Metrics (used by all baselines)
- **Hits@1**: Is the correct answer the top-ranked answer? (primary metric)
- **Hits@5**: Is correct answer in top 5?
- **F1**: Token-level F1 between predicted and gold answer
- **MRR**: Mean Reciprocal Rank

### New Temporal-Specific Metrics [CONTRIBUTION]

**Temporal Grounding Score (TGS)**:
> What fraction of retrieved triples are temporally valid for the question's time window?
```
TGS = |{t in retrieved : t.t_start ≤ t_anchor ≤ t.t_end}| / |retrieved|
```
Measures retrieval quality from a temporal perspective, not just answer quality.

**Temporal Operator Accuracy (TOA)**:
> Per-operator-type Hits@1 breakdown (BEFORE, AFTER, DURING, FIRST, LAST)
Reveals where the system fails (e.g., "FIRST/LAST" questions are hardest).

**Temporal Consistency Rate (TCR)**:
> Among correct answers, what fraction are verified by TCV?
Measures alignment between LLM generation and KG temporal facts.

---

## 7. EXPECTED CONTRIBUTIONS

### C1 — Technical System
A fully training-free temporal subgraph RAG pipeline for TKGQA, requiring
no GPU training, runnable on consumer hardware (M3 Max / any 16GB+ machine).

### C2 — Novel Temporal Relevance Score (TRS)
A principled, interpretable scoring function for ranking KG triples by
temporal relevance to a given question. Three interpretable components
(proximity, operator match, entity relevance) with ablation study.

### C3 — New Evaluation Metrics
Three new temporal-specific metrics: TGS, TOA, TCR. These expose failure
modes invisible to Hits@1 alone and will benefit future TKGQA research.

### C4 — Comprehensive Benchmark
First unified evaluation of 8 methods (2 LLM-only, 2 static RAG, 2 classic,
2 modern LLM-based) on 3 standard temporal KGQA datasets with unified
preprocessing and evaluation code — all released as open source.

### C5 — Efficiency Analysis
First paper to report wall-clock latency, memory usage, and cost-per-query
for temporal KGQA methods on consumer hardware — enabling future work
without GPU infrastructure.

---

## 8. EXPECTED RESULTS (HYPOTHESIS)

On CronQuestions test set (Hits@1):

| Method | Expected Hits@1 | Notes |
|--------|----------------|-------|
| LLM-Direct | ~0.15 | Hallucination-heavy |
| BM25-RAG | ~0.25 | Temporally blind retrieval |
| Dense-RAG | ~0.28 | Better recall, still temporally blind |
| CRONKGQA | ~0.46 | Published number |
| TempoQR | ~0.49 | Published number |
| GenTKGQA | ~0.55 | Published number |
| RTQA | ~0.58 | Published number |
| **TempoRAG (ours)** | **~0.52-0.57** | Competitive, training-free |

Key claim: TempoRAG achieves competitive performance with *zero training*,
while being 10-50x faster and requiring no GPU.

---

## 9. PROJECT STRUCTURE

```
rag/
├── docs/
│   ├── PAPER_PLAN.md          ← this file
│   └── RELATED_WORK.md
├── src/
│   ├── stage1/
│   │   ├── entity_linker.py   ← entity detection + KG linking
│   │   ├── time_extractor.py  ← temporal anchor extraction
│   │   └── op_classifier.py   ← temporal operator classification
│   ├── stage2/
│   │   ├── subgraph.py        ← k-hop subgraph extraction
│   │   ├── trs_scorer.py      ← Temporal Relevance Score
│   │   └── retriever.py       ← top-N triple selection
│   ├── stage3/
│   │   ├── prompt_builder.py  ← structured prompt construction
│   │   ├── llm_client.py      ← Ollama API wrapper
│   │   └── tcv.py             ← Temporal Constraint Verifier
│   ├── metrics/
│   │   ├── standard.py        ← Hits@K, F1, MRR
│   │   └── temporal.py        ← TGS, TOA, TCR
│   └── baselines/
│       ├── llm_direct.py
│       ├── bm25_rag.py
│       ├── dense_rag.py
│       └── cronkgqa_wrapper.py
├── experiments/
│   ├── run_tempoRAG.py
│   ├── run_baselines.py
│   └── ablations.py
├── data/
│   ├── cronquestions/
│   ├── timequestions/
│   └── multitq/
├── results/
└── README.md
```

---

## 10. IMPLEMENTATION PLAN (PHASES)

### Phase 1 — Data & Environment (Week 1-2)
- [ ] Download CronQuestions, TimeQuestions, MultiTQ
- [ ] Build KG graph objects (NetworkX) for Wikidata subset + ICEWS14
- [ ] Install: spaCy, ollama (Llama-3.1-8B-Q4), PyKEEN (TComplEx), FAISS
- [ ] Implement evaluation metrics (Hits@1, F1, MRR, TGS, TOA, TCR)

### Phase 2 — Baseline Implementation (Week 3-4)
- [ ] LLM-Direct + LLM-CoT
- [ ] BM25-RAG + Dense-RAG
- [ ] CRONKGQA (official code integration)
- [ ] TempoQR (official code integration)

### Phase 3 — TempoRAG Implementation (Week 5-7)
- [ ] Stage 1: entity_linker + time_extractor + op_classifier
- [ ] Stage 2: subgraph extractor + TRS scorer
- [ ] Stage 3: prompt_builder + llm_client + TCV
- [ ] End-to-end pipeline test

### Phase 4 — Experiments (Week 8-10)
- [ ] Run all baselines on all 3 datasets
- [ ] Run TempoRAG + all ablations
- [ ] Collect latency + memory stats
- [ ] Statistical significance testing

### Phase 5 — Analysis & Writing (Week 11-14)
- [ ] Error analysis (sample 100 failures per method)
- [ ] Case studies (qualitative examples)
- [ ] Figure generation
- [ ] Paper writing

---

## 11. HARDWARE REQUIREMENTS

| Component | Memory | Notes |
|-----------|--------|-------|
| Llama-3.1-8B-Q4 | ~5GB | via Ollama |
| TComplEx embeddings (125K entities) | ~1.5GB | pre-trained, frozen |
| NetworkX KG (Wikidata subset) | ~2-4GB | |
| FAISS index | ~500MB | |
| Working memory | ~5GB | |
| **Total** | **~15GB** | Well within 36GB |

---

## 12. NOVELTY CHECKLIST

- [x] Training-free (no GPU training required)
- [x] Novel TRS scoring function (3-component, interpretable)
- [x] New evaluation metrics (TGS, TOA, TCR)
- [x] First unified benchmark across 8 methods × 3 datasets
- [x] First efficiency analysis for TKGQA on consumer hardware
- [x] Temporal Constraint Verifier (TCV) post-hoc grounding
- [x] Fully reproducible (all code + data released)
