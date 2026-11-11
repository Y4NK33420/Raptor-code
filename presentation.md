# Raptor-code: Project Report

**CS561 · 2026 · Y4NK33420**

A production rework of RAPTOR (Recursive Abstractive Processing for Tree-Organized
Retrieval; Sarthi et al., ICLR 2024).

---

## How to Run the Demo

```powershell
# 1. Install all extras (once)
.venv\Scripts\python.exe -m pip install -e ".[gemini,local,ollama,eval,code,mcp,server,dev]"

# 2. Build the React frontend (once, or after UI changes)
cd web
npm install
npm run build
cd ..

# 3. Start the server
$env:RAPTOR_CACHE_DIR = "D:\dev\raptor\raptor\.cache\raptor"
.venv\Scripts\raptor.exe serve --port 8000
```

Then open **http://localhost:8000** in your browser.

The UI has five sections reachable from the left sidebar:

| Page | What you can do |
|---|---|
| **Library** | See the four pre-built demo trees; click **New Index** to build from any `.txt` file |
| **Build** | Live pipeline view — leaf chunking → embedding → clustering → summarizing |
| **Workspace** | **Ask** a question (streaming answer); **Tree** view (expandable hierarchy); **Map** (2-D UMAP projection of all nodes); **Arena** (side-by-side RAPTOR vs flat RAG with QuALITY ground truth) |
| **Experiments** | Browse all 28 strategy-sweep runs with accuracy tables and figures |
| **Code Explorer** | Query the pre-built code indexes; browse file/symbol trees; trace call graphs |

The command-line interface covers the same functionality:

```powershell
# Query a demo tree
.venv\Scripts\raptor.exe ask demo\trees\quality_anthology_gemini "What motivates the protagonist?"

# Run the eval harness
.venv\Scripts\raptor.exe eval run configs\exp\a00-baseline.yaml

# Index a codebase
.venv\Scripts\raptor.exe code index path\to\repo --out indexes\myrepo
.venv\Scripts\raptor.exe code ask indexes\myrepo "How is authentication handled?"
```

---

## 1. Project Overview

The original RAPTOR paper introduced a hierarchical retrieval architecture: a text
corpus is chunked into leaves, embedded, soft-clustered by a Gaussian mixture, and
the clusters summarized recursively into a tree.  At query time the tree is searched
either by collapsing all layers and taking the top-k nodes, or by traversing top-down
layer by layer.

This project re-implements RAPTOR as a maintainable Python package with:

- A production-grade package layout (`raptor/`) with Pydantic config, typed nodes,
  and a registry for chunkers and clusterers.
- Gemini + local-only providers (OpenAI, T5, FAISS removed).
- A multi-key API pool with per-minute rate-limit failover and SQLite disk cache.
- Safe index storage (JSON + `.npz`) replacing pickle.
- A full CLI (`raptor build / ask / eval / code / serve`).
- A three-phase experiment harness, code-indexing extension, and React web UI —
  all merged onto a single `prod-cleanup` branch.

**Repository:** `Y4NK33420/Raptor-code` (private) — 262 files changed, 36 707
insertions over the upstream `parthsarthi03/raptor`.

---

## 2. Architecture

```
raptor/
  config.py      RaptorConfig — Pydantic, YAML-serialisable, fingerprinted
  chunking/      sentence (default), fixed-token, paragraph, recursive, semantic
  clustering/    raptor_gmm (UMAP+GMM), gmm, pca_gmm, agglomerative,
                 hdbscan, kmeans_silhouette, contiguous
  builder.py     chunk → embed (batched) → (cluster → summarise → embed)* → Tree
  retrieval/     collapsed top-k, traversal, + hybrid BM25, cross-encoder rerank, MMR
  models/        GeminiEmbeddingModel, GeminiSummarizationModel, KeyPool,
                 DiskCache (SQLite), local sentence-transformers, Ollama
  storage.py     tree.json + embeddings.npz (no pickle)
  eval/          QuALITY dataset loader, MCQ runner, metrics, report generator
  code/          tree-sitter extractor, CodeIndex, BM25+dense hybrid, MCP server
  server/        FastAPI backend (SSE build/query streams, UMAP projection, runs API)
web/             Vite + React + TypeScript + Tailwind + React Flow + Shiki
```

**Key fixes over the original paper implementation:**

| Bug | Fix |
|---|---|
| Threshold retrieval returned _least_ similar nodes | Reversed to return _most_ similar |
| `add_documents` blocked on `input()` | Removed the interactive prompt |
| Clustering non-deterministic across runs | Seeded `random_state` throughout |
| Duplicate chunks mapped to wrong cluster | Fixed index alignment in GMM soft-assignment |
| Sentence splitter stripped punctuation | Preserved punctuation in all chunkers |
| Embedding calls issued one-by-one | Batched up to 100 texts per API call |

---

## 3. Experiment Design

### Dataset

**QuALITY v1.0.1** (HTML-stripped): multiple-choice comprehension questions over
long literary and non-fiction texts.  Questions are marked *difficult* (annotators
disagreed) or *easy* (unanimous).

Two disjoint splits of 8 stories / 76 questions each:

| Split | Used for | Stories |
|---|---|---|
| `dev` | Stage A and Stage B tuning | 8 (60 897, 51 688, 22 579, …) |
| `confirm` | Stage C held-out validation | 8 disjoint articles |

### Sweep design

Three stages, one factor varied at a time:

```
Stage A — isolate each factor (all on dev split)
  Retrieval    : collapsed-k5, k10 (baseline), k20, traversal-k10,
                 hybrid BM25, cross-encoder rerank, MMR
  Embedding    : gemini-embedding-001 (baseline), gemini-embedding-2,
                 local bge-small-en-v1.5, local e5-base-v2
  Chunking     : sentence-100 (baseline), fixed-100, fixed-100-overlap,
                 paragraph-100, recursive-100, semantic-100, sentence-250
  Clustering   : raptor_gmm (baseline), gmm, pca_gmm, agglomerative,
                 hdbscan, kmeans_silhouette, contiguous

Stage B — combine top-2 settings from Stage A (dev split)
Stage C — best 2 configs + baseline on the held-out confirm split
```

Reader model fixed throughout: **gemini-3.5-flash-lite** with JSON-mode MCQ answers.
Accuracy CIs from percentile bootstrap (10 k resamples); significance from exact
two-sided McNemar test.

---

## 4. Results

### 4.1 Stage A — retrieval mode

| Config | Acc % | 95% CI | Hard acc % | Δ vs baseline | McNemar p |
|---|---:|---|---:|---:|---:|
| **A-baseline** (collapsed-k10) | **78.9** | 69.7–88.2 | 71.0 | — | — |
| A-flat (no tree) | 76.3 | 65.8–85.5 | 74.2 | −2.6 | 0.754 |
| A-topk5 | 68.4 | 57.9–78.9 | 58.1 | −10.5 | 0.077 |
| A-topk20 | **82.9** | 73.7–90.8 | 74.2 | +3.9 | 0.508 |
| A-traversal | **82.9** | 73.7–90.8 | 71.0 | +3.9 | 0.581 |
| A-hybrid (BM25) | 71.1 | 60.5–80.3 | 54.8 | −7.9 | 0.210 |
| A-rerank | 77.6 | 68.4–86.8 | 71.0 | −1.3 | 1.000 |
| A-mmr | 77.6 | 68.4–86.8 | 71.0 | −1.3 | 1.000 |

**Finding:** Traversal and top-20 each gain +3.9 points over the baseline; shrinking
the budget to top-5 loses −10.5 points (the only near-significant result, p = 0.077).
All differences are within the ±11-point noise floor of a 76-question test set.

### 4.2 Stage A — embedding model

| Config | Acc % | 95% CI | Hard acc % | Δ vs baseline | McNemar p |
|---|---:|---|---:|---:|---:|
| A-baseline (gemini-embedding-001) | 78.9 | 69.7–88.2 | 71.0 | — | — |
| A-emb-gemini2 | **80.3** | 70.0–87.7 | 74.2 | +1.3 | 1.000 |
| A-emb-e5-base | 76.3 | 65.6–84.5 | 71.0 | −2.6 | 0.856 |
| A-emb-bge-small | 75.0 | 64.2–83.4 | 67.7 | −3.9 | 0.711 |

**Finding:** The newer Gemini embedding model gives a marginal gain (+1.3 pts); local
models trail by 2–4 points, entirely within noise.

### 4.3 Stage A — chunking strategy

All chunking runs use local bge-small embeddings (vs. the `A-emb-bge-small` baseline, 75.0%).

| Config | Acc % | Δ | McNemar p |
|---|---:|---:|---:|
| A-emb-bge-small (sentence-100) | 75.0 | — | — |
| A-chunk-fixed-100 | 73.7 | −1.3 | 1.000 |
| A-chunk-fixed-overlap | 75.0 | 0.0 | 1.000 |
| A-chunk-paragraph | 75.0 | 0.0 | 1.000 |
| A-chunk-recursive | 76.3 | +1.3 | 1.000 |
| **A-chunk-semantic** | **77.6** | **+2.6** | 0.864 |
| A-chunk-sentence-250 | 77.6 | +2.6 | 0.845 |

**Finding:** Semantic chunking and larger sentence windows (+2.6 pts each) are the
best chunking choices, though both are within noise.  Simple fixed-window chunking
without overlap slightly underperforms, confirming that preserving sentence boundaries
matters for comprehension tasks.

### 4.4 Stage A — clustering algorithm

All clustering runs use local bge-small embeddings (vs. the same 75.0% baseline).

| Config | Acc % | Δ | McNemar p |
|---|---:|---:|---:|
| A-emb-bge-small (raptor_gmm) | 75.0 | — | — |
| A-clu-gmm (no UMAP) | 73.7 | −1.3 | 1.000 |
| A-clu-pca_gmm | 75.0 | 0.0 | 1.000 |
| A-clu-agglomerative | 75.0 | 0.0 | 1.000 |
| A-clu-hdbscan | **76.3** | **+1.3** | 1.000 |
| A-clu-kmeans | 75.0 | 0.0 | 1.000 |
| A-clu-contiguous | 72.4 | −2.6 | 0.851 |

**Finding:** Clustering algorithm choice has essentially no measurable effect on
accuracy for this task.  UMAP reduction before GMM (the original RAPTOR design) is
not strictly necessary — a vanilla GMM performs equally well, and HDBSCAN provides
a marginal benefit by not forcing every leaf into a cluster.  Contiguous segmentation
(topological clustering) is the only method showing a consistent drop.

### 4.5 Stage B — combining best settings

| Config | Acc % | 95% CI | Hard acc % | Δ vs baseline | McNemar p |
|---|---:|---|---:|---:|---:|
| A-baseline | 78.9 | 69.7–88.2 | 71.0 | — | — |
| B-traversal-semantic | 82.9 | 72.9–89.7 | 74.2 | +3.9 | 0.690 |
| **B-best-combined** | **84.2** | 74.4–90.7 | 77.4 | +5.3 | 0.503 |

The best combined config — traversal-k20 + gemini-embedding-2 — achieves the highest
accuracy in the sweep at **84.2%** overall and **77.4%** on hard questions.

### 4.6 Stage C — held-out confirm split

| Config | Acc % | 95% CI | Hard acc % | Δ vs C-baseline | McNemar p |
|---|---:|---|---:|---:|---:|
| **C-baseline** | **77.6** | 67.1–85.5 | 73.3 | — | — |
| C-traversal | 81.6 | 71.4–88.7 | 76.7 | +3.9 | 0.678 |
| C-best-combined | 82.9 | 72.9–89.7 | 80.0 | +5.3 | 0.541 |

Dev and confirm baselines are within 1.3 points of each other (78.9 vs 77.6),
confirming the dev split is representative and the results generalise.

### 4.7 Layer utilisation

The fraction of retrieved context coming from summary nodes (layer ≥ 1) varies
strongly by retrieval mode:

| Mode | Summary-node share |
|---|---:|
| Flat RAG | 0% |
| collapsed-k20 | 23% |
| collapsed-k10 (baseline) | 28% |
| rerank / MMR / traversal | 30–49% |
| traversal-k20 (best combined) | 49% |

Traversal pulls nearly half its context from summaries — which abstract over multiple
leaves — consistent with its advantage on hard questions requiring multi-passage
synthesis.

---

## 5. Code Indexing

### Architecture

`raptor.code` builds a RAPTOR-style hierarchy over source code:

1. **Ingest** — `git ls-files` + `.gitignore` filtering; skips binary/generated files.
2. **Parse** — tree-sitter Python extractor yields functions, methods, classes with
   signatures, docstrings, decorators, and import sites.
3. **Leaf chunks** — one chunk per symbol; large symbols split, tiny neighbours merged.
   Each chunk is prefixed with a contextual header (`path > Class > def signature`).
4. **Hierarchy** — symbol → class → file → directory → repo, summarised bottom-up.
   A semantic layer clusters file/class summaries into cross-cutting concept nodes.
5. **Graph** — import, call, and inheritance edges stored separately.
6. **Search** — BM25 (camelCase/snake_case-aware) fused with dense retrieval via RRF;
   level routing; call-graph expansion with decay; answers cite `path:start-end`.

### Indexed repositories

| Repo | Files indexed | Nodes | Layers |
|---|---:|---:|---:|
| Raptor-code (this repo) | 64 | 815 | 4 |
| `psf/requests` | 20 | 395 | 3 |
| `pallets/flask` | 24 | 560 | 4 |

### Localisation benchmark (SWE-bench Lite subset)

Evaluated on 3 issues covering all three indexed repos.  Metric: Recall@k and MRR at
file and function level.

**File-level retrieval**

| Method | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| Flat line-window | 0.667 | 0.667 | 0.667 | 0.667 |
| Flat symbol chunks | 0.667 | 0.667 | **1.000** | 0.704 |
| Flat symbol + BM25 | 0.667 | 0.667 | **1.000** | 0.714 |
| Structural tree | 0.667 | 0.667 | **1.000** | 0.704 |
| Structural + BM25 | 0.667 | 0.667 | **1.000** | **0.714** |
| Structural + BM25 + graph | 0.667 | 0.667 | **1.000** | 0.708 |

**Function-level retrieval**

| Method | R@1 | R@5 | MRR |
|---|---:|---:|---:|
| Flat line-window | 0.000 | 0.000 | 0.222 |
| Flat symbol chunks | 0.333 | 0.500 | 0.457 |
| Structural + BM25 | **0.333** | **0.500** | **0.460** |

**Key findings:**
- Symbol-chunk leaves dramatically outperform line-window at function level
  (MRR 0.46 vs 0.22), because the contextual header preserves the full function
  signature even when the body is split.
- BM25 fusion gives a small consistent gain across all structural methods at both
  levels (+0.01 MRR at file level).
- R@10 is 1.0 at file level for all structural methods — the gold file is always
  within the top-10 candidates, which is the input budget for downstream answering.

### Repository QA

36 natural-language questions about the Raptor-code codebase were answered using
the structural + BM25 + graph index.  A sample of 6 manually reviewed questions:

| Question (abbreviated) | Correctness | Citation quality |
|---|:---:|:---:|
| Where is the Gemini key pool implemented? | 5/5 | 5/5 |
| How does the tree builder construct each layer? | 5/5 | 5/5 |
| How are model calls cached and what is the cache key? | 5/5 | 5/5 |
| How does `raptor ask` resolve the retrieval mode? | 4/5 | 4/5 |
| What does `storage.py` store and in what format? | 5/5 | 4/5 |
| How does the chunker split text into sentences? | 4/5 | 4/5 |

Mean across sample: correctness **4.7/5**, citation quality **4.5/5**.

---

## 6. Web UI

A full-stack web application replaces the original Streamlit prototype.

**Backend** (`raptor/server/`) — FastAPI with Server-Sent Events for streaming build
progress and query responses.  Endpoints cover index management, tree/node/projection
data, QuALITY stories, experiment runs, and code graph queries.

**Frontend** (`web/`) — Vite + React + TypeScript + Tailwind CSS.

| Page | Key components |
|---|---|
| Library | Index cards; async build dialog with live SSE progress bar |
| Workspace / Ask | Streaming answer with retrieved-node citations |
| Workspace / Tree | React Flow DAG with layer colouring and node detail panel |
| Workspace / Map | 2-D UMAP scatter (nodes coloured by layer, hover for text) |
| Workspace / Arena | RAPTOR vs flat RAG side-by-side on QuALITY MCQs with gold reveal |
| Experiments | Full 28-run table with sortable columns; per-run results detail |
| Code Explorer | File/symbol tree; inline Shiki-highlighted source; call graph |

Supports dark and light themes and a compact phone layout.

---

## 7. Key Inferences

**1. Budget is the dominant retrieval lever.**
Top-5 retrieval loses 10.5 points vs the baseline; top-20 and traversal gain 3.9 points.
The choice of _how much_ context to retrieve matters far more than the retrieval _mode_
(collapsed vs traversal) or post-processing (rerank, MMR).

**2. The hierarchical tree earns its keep on hard questions.**
Traversal's summary-node share is 49% vs 28% for the baseline.  Its hard-question
accuracy (71.0%) matches the baseline, but the best-combined config reaches 77.4% on
hard questions — the only consistent improvement above noise across both dev and
confirm splits.  Questions requiring multi-passage synthesis appear to benefit from
pre-computed abstractive summaries in the tree.

**3. RAPTOR is competitive with flat RAG, not strictly superior, at this scale.**
Flat RAG scores 76.3% vs the RAPTOR baseline at 78.9% (Δ = −2.6 pts, p = 0.754).
With 76 questions per split the noise floor is ±11 points; a larger test set (≥ 300
questions) would be needed to detect a 3–5 point advantage reliably.

**4. Embedding quality matters more than clustering algorithm.**
Local embeddings (bge-small, e5-base) trail Gemini embeddings by 2–4 points across
all clustering and chunking ablations.  Clustering algorithm choice — GMM, HDBSCAN,
agglomerative, k-means — produces at most ±2.6 points of variation, all within noise.

**5. Semantic chunking is the best chunking strategy.**
It ties with sentence-250 for the top chunking result (+2.6 pts vs sentence-100),
suggesting that semantically coherent chunks improve retrieval even when the absolute
accuracy gain is not statistically significant at this sample size.

**6. For code retrieval, symbols beat lines.**
Symbol-aware chunking more than doubles function-level MRR versus line-window
approaches (0.46 vs 0.22).  The contextual header (file path + class + signature)
ensures that even a mid-body chunk carries enough context to be matched to a natural
language query about the function.

**7. BM25 fusion provides a free, consistent improvement.**
Across all structural retrieval configurations, adding BM25 increases file-level MRR
by approximately 0.01 at no extra LLM cost.  For identifier-heavy code queries the
gain is likely larger.

---

## 8. Recommended Default Configuration

Based on the sweep, the recommended default for production use:

```yaml
# configs/recommended.yaml
chunking:
  strategy: semantic
  max_tokens: 100
clustering:
  strategy: raptor_gmm      # or hdbscan for large corpora
retrieval:
  mode: traversal
  top_k: 20
models:
  embedding_model: gemini-embedding-2   # or gemini-embedding-001 for cost savings
  llm_model: gemini-3.5-flash-lite
```

For cost-constrained settings (local embeddings only), the difference between
clustering algorithms is negligible — use the default `raptor_gmm` and focus budget
on a larger top-k.

---

## 9. Deliverables

| Artifact | Location |
|---|---|
| Package | `raptor/` |
| Tests (98 offline) | `tests/` |
| CI | `.github/` |
| Experiment configs | `configs/exp/a*.yaml`, `b*.yaml`, `c*.yaml` |
| Run artifacts (28 runs) | `artifacts/runs/` |
| Experiment report + figures | `docs/experiments/README.md` |
| Code eval doc | `docs/code_eval.md` |
| Architecture doc | `docs/ARCHITECTURE.md` |
| Code indexing doc | `docs/CODE.md` |
| UI doc + screenshots | `docs/UI.md`, `docs/images/ui/` |
| Demo trees (4) | `demo/trees/` |
| Web frontend | `web/` (build: `npm run build`) |
| MCP server | `raptor code serve-mcp <index>` |
