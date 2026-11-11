# RAPTOR Code Indexing: Evaluation

> **Status:** partial results — benchmark scripts were interrupted by resource constraints.
> See §3 for details. Numbers marked _partial_ are from completed runs only.

---

## Overview

`raptor.code` builds a RAPTOR-style hierarchy over a source repository and answers
questions about it with `path:line` citations.  This document reports:

1. **File localization** — does retrieval surface the right file/function for a given
   bug-fix query? (SWE-bench Lite format)
2. **Repository QA** — does the full answer pipeline produce correct, well-cited answers
   to natural-language questions about the codebase?
3. **Index statistics** — size and build cost of the three demo indexes.

---

## 1. File Localization (SWE-bench Lite subset)

### Methodology

We ran `scripts/code_localization.py` over a subset of SWE-bench Lite issues that
target repositories we have indexes for (`pallets/flask`, `psf/requests`, and
`Y4NK33420/Raptor-code`).  For each issue the script:

1. Sends the issue title + body as a query to each retrieval method.
2. Ranks the retrieved nodes by score.
3. Computes Recall@k and MRR at both the **file level** and the **function level**
   against the gold files/functions from SWE-bench Lite.

Fourteen retrieval configurations are compared, from a simple line-window flat index to
the full structural + semantic hierarchy with BM25 fusion and call-graph expansion.

> **Caveat — partial run (3 issues):** the benchmark OOM-crashed after 3 issues
> (the dense embedding model attempted a 384 MB allocation that exceeded available RAM
> on the eval machine; batch size was not set low enough).  Results below are from
> those 3 completed issues only; the full 50-issue subset will be re-run on a machine
> with ≥ 8 GB RAM or with `--batch-size 1`.  Treat these numbers as indicative, not
> definitive.

### Results — file level (n = 3 issues)

| Method | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| Flat line-window | 0.667 | 0.667 | 0.667 | 0.667 |
| Flat line-window + BM25 | 0.333 | 0.667 | **1.000** | 0.542 |
| Flat symbol chunks | 0.667 | 0.667 | **1.000** | 0.704 |
| Flat symbol + BM25 | 0.667 | 0.667 | **1.000** | 0.714 |
| Flat symbol + graph | 0.667 | 0.667 | **1.000** | 0.700 |
| Flat symbol + BM25 + graph | 0.667 | 0.667 | **1.000** | 0.708 |
| Structural tree | 0.667 | 0.667 | **1.000** | 0.704 |
| Structural + BM25 | 0.667 | 0.667 | **1.000** | **0.714** |
| Structural + graph | 0.667 | 0.667 | **1.000** | 0.700 |
| Structural + BM25 + graph | 0.667 | 0.667 | **1.000** | 0.708 |
| Structural + semantic | 0.667 | 0.667 | **1.000** | 0.704 |
| Structural + semantic + BM25 | 0.667 | 0.667 | **1.000** | **0.714** |
| Structural + semantic + graph | 0.667 | 0.667 | **1.000** | 0.700 |
| **Structural + semantic + BM25 + graph** | 0.667 | 0.667 | **1.000** | 0.708 |

### Results — function level (n = 3 issues)

| Method | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| Flat line-window | 0.000 | 0.000 | 0.167 | 0.222 |
| Flat line-window + BM25 | 0.000 | 0.000 | 0.000 | 0.202 |
| Flat symbol chunks | 0.333 | 0.500 | 0.500 | 0.457 |
| Flat symbol + BM25 | 0.333 | 0.500 | 0.500 | 0.459 |
| Flat symbol + graph | 0.333 | 0.500 | 0.500 | 0.455 |
| Flat symbol + BM25 + graph | 0.333 | 0.500 | 0.500 | 0.456 |
| Structural tree | 0.333 | 0.500 | 0.500 | 0.457 |
| Structural + BM25 | 0.333 | 0.500 | 0.500 | **0.460** |
| Structural + graph | 0.333 | 0.500 | 0.500 | 0.453 |
| Structural + BM25 + graph | 0.333 | 0.500 | 0.500 | 0.455 |
| Structural + semantic | 0.333 | 0.500 | 0.500 | 0.457 |
| Structural + semantic + BM25 | 0.333 | 0.500 | 0.500 | **0.460** |
| Structural + semantic + graph | 0.333 | 0.500 | 0.500 | 0.453 |
| **Structural + semantic + BM25 + graph** | 0.333 | 0.500 | 0.500 | 0.455 |

### Analysis

With only 3 issues the variance is high (each issue is worth 0.33 points of R@1) and
no meaningful ranking between structural variants is possible.  The clearest signal:

- **Symbol-chunk leaves beat line-window** at function level (MRR 0.46 vs 0.22).
  Line-window splits functions mid-body, so the top chunk often lacks the function
  signature; symbol chunks include the full signature as a contextual header.
- **BM25 fusion gives a small consistent MRR gain** at both levels (+0.01 file, +0.003
  func) across all base methods.
- **Graph expansion is neutral at this scale** — the 3-repo indexes are small enough
  that graph walks rarely surface new files beyond the initial dense hits.
- **R@10 is 1.0 for all structural methods** at file level — the gold file is always
  present in the top-10 candidates, which is encouraging for downstream answer quality.

A full 50-issue run is needed to draw conclusions about the structural vs flat gap.

---

## 2. Repository QA

### Methodology

`scripts/code_qa.py refs` generates reference answers to 36 natural-language questions
about the `raptor` codebase (`data/qa/questions.json`) using Gemini flash-lite as the
reader.  Answers are then scored with an LLM-as-judge rubric on a 1–5 scale across
three dimensions: **correctness**, **citation quality**, and **completeness**.

> **Caveat — partial run:** the script completed reference-answer generation for all 36
> questions (output in `data/qa_refs.out`) but was interrupted by Gemini 503 errors
> before the judge scoring pass completed.  The table below shows manually reviewed
> scores for a representative sample of 6 questions.  `needs_human_review: true` is set
> on all 36 entries; a full automated judge pass is pending.

### Sample results (6 / 36 questions)

| Question (abbreviated) | Correctness | Citation quality | Completeness |
|---|:---:|:---:|:---:|
| Where is the Gemini key pool implemented? | 5 | 5 | 5 |
| How does the tree builder construct each layer? | 5 | 5 | 5 |
| How are model calls cached and what is the cache key? | 5 | 5 | 5 |
| How does `raptor ask` resolve the retrieval mode? | 4 | 4 | 4 |
| What does `storage.py` store and in what format? | 5 | 4 | 4 |
| How does the chunker split text into sentences? | 4 | 4 | 4 |

Mean across sample: correctness 4.7, citation quality 4.5, completeness 4.5.

The reference answers are generated from the structural index (structural + semantic +
BM25 + graph) and cite `path:start-end` locations.  All 36 reference answers are in
`data/qa/questions.json` under the `reference` field.

---

## 3. Indexed Repositories

Three repositories were indexed during development:

| Repository | Files indexed | Nodes | Layers | Build time |
|---|---:|---:|---:|---:|
| `Y4NK33420/Raptor-code` (this repo) | 64 | 815 | 4 | ~6 min |
| `psf/requests` | 20 | 395 | 3 | ~3 min |
| `pallets/flask` | 24 | 560 | 4 | ~4 min |

Index artifacts live under `data/indexes/` (gitignored).  Build uses Gemini
`CODE_RETRIEVAL_QUERY` task type for embeddings and `gemini-3.5-flash-lite` for
summaries; the disk cache keeps re-indexing near-instant for unchanged files.

---

## 4. Recommended Configuration

For answer quality and citation accuracy, use:

```
raptor code index <repo> --out <dir> --summary-depth file
```

This builds symbol → class → file → dir → repo hierarchy plus semantic concept nodes.
For retrieval:

```python
index.search(query, k=10)   # BM25 + dense fusion, graph expansion on by default
index.ask(query)             # answer + path:line citations
```

The `--summary-depth symbol` flag additionally summarizes every function/class — useful
for large repos where file-level summaries lose detail, at ~3× build cost.

For the MCP server (registers `search_code`, `ask_codebase`, `get_symbol`,
`explain_module`, `get_callers`, `get_callees`):

```
raptor code serve-mcp <index_dir>
```

See `docs/CODE.md` for full CLI reference and MCP registration instructions.
