# RAPTOR: production cleanup → strategy experiments → code-repo extension

## Context
This repo is a fork of upstream RAPTOR (`parthsarthi03/raptor`), extended for a CS561 project with Gemini models, a Flat RAG baseline, a Streamlit dashboard, and QuALITY scripts. It runs, but it isn't production quality:
- a hardcoded API key
- `pickle` tree storage
- `input()` calls inside library code
- OpenAI clients built eagerly as defaults
- several real bugs in chunking, clustering and retrieval
- no tests or packaging
- "results" that are hardcoded text

The goal has four parts:
1. Make the package clean, tested and installable.
2. Build a cheap, reproducible experiment harness and use it to compare chunking, clustering, retrieval and embedding strategies on QuALITY.
3. Extend RAPTOR to codebases, with a CLI and an MCP server.
4. Replace the Streamlit dashboard with a new React UI backed by a FastAPI service.

The work moves to your private repo `Y4NK33420/Raptor-code`, crediting the original RAPTOR authors.

**Decisions made:**
- Restructure into subpackages, keeping a compatibility layer so `from raptor import RetrievalAugmentation` still works.
- Gemini plus local models. Drop the OpenAI and T5 code.
- Free-tier budget, so experiments use 5–10 stories, heavy caching, and local models where possible.
- Code support: Python first (built on tree-sitter so other languages can follow), exposed through a CLI and an MCP server.

---

## Phase 0: Hygiene and safety (first, small)
- **Remotes:** rename the current `origin` (upstream `parthsarthi03/raptor`) to `upstream`, and add `https://github.com/Y4NK33420/Raptor-code.git` as `origin`. Work on a `prod-cleanup` branch; push only when you say so.
- **Credit:** keep `LICENSE.txt` (MIT, original copyright) and add your own copyright line. Add a README "Acknowledgements" section that cites the RAPTOR paper (Sarthi et al., ICLR 2024) and links the upstream repo.
- **Key:** load it only from `.env` or the environment (pydantic-settings), so it never gets committed, even to the private repo. Delete the hardcoded value, the hand-written `.env` parser and the Windows registry lookup in `run_gemini.py`. Your existing key goes into a local `.env`; rotation is deferred, as you asked.
- Add to `.gitignore`: `.venv*`, `.venv.copied-from-other-machine/`, the PDF, `data/`, `artifacts/`, `.cache/`, `quality_repo/` (213 MB, its own git repo). Replace `quality_repo/` with `scripts/download_quality.py`, which fetches `QuALITY.v1.0.1.htmlstripped.*` into `data/quality/`.
- Move the one-off root scripts out of the root: `test_diff.py`, `build_story2.py`, `compare_baseline.py`, `test_quality.py`, `run_gemini.py`, `generate_plots.py`. Each goes to `examples/`, gets superseded by the eval harness (Phase 2), or is deleted.

## Phase 1: Production cleanup
### 1a. Target layout (pluggable strategies via small ABCs plus a name registry)
```
raptor/
  __init__.py          # public API + compat re-exports (RetrievalAugmentation, Config, FlatRAG, Gemini*)
  config.py            # pydantic models: RaptorConfig{chunking, clustering, builder, retrieval, models}; YAML load/dump
  settings.py          # env/.env settings (GEMINI_API_KEY, cache dir, rate limits)
  types.py             # Node/Tree as dataclasses (+ metadata: source, span, layer, parents) — DAG-capable
  models/  base.py (Embedder/Summarizer/QA protocols, batch API)
           gemini.py   (google-genai SDK, batch embed, task_type doc/query/code, token-bucket limiter, tenacity)
           local.py    (sentence-transformers embedder, cross-encoder reranker)
           ollama.py   (optional local LLM summarizer/QA to dodge free-tier limits)
           cache.py    (SQLite disk cache wrapper keyed by model+params+text hash)
  chunking/   base.py, sentence.py (fixed port of split_text), registry
  clustering/ base.py, raptor_gmm.py (fixed port of cluster_utils), registry
  builder.py           # merges tree_builder.py + cluster_tree_builder.py
  retrieval/  collapsed.py, traversal.py, flat.py (FlatRAG), base.py
  storage.py           # versioned dir format: tree.json + embeddings.npy; legacy pickle import behind trusted=True
  pipeline.py          # RetrievalAugmentation (no input(), add_documents(overwrite=...), add_to_existing)
  cli.py               # `raptor build/ask` (code commands added in Phase 3)
  server/              # FastAPI service layer used by the React UI (Phase 4); stays thin over pipeline/eval/code
web/                   # React app (Phase 4)
```
The Streamlit `app.py` stays runnable until the React UI covers everything it does, then it's deleted.
Delete `QAModels.py`, `SummarizationModels.py` (OpenAI/T5), `EmbeddingModels.OpenAIEmbeddingModel`, and `FaissRetriever.py`. FAISS returns later as an optional backend only if it's needed for code-scale indexes.

### 1b. Bugs to fix (each gets a regression test)
**`utils.split_text`** ([utils.py](raptor/utils.py))
- The split regex deletes `.`, `!` and `?`, so chunks lose their sentence punctuation.
- The overlap token count indexes the global `n_tokens` with chunk-local positions.

**`cluster_utils`** ([cluster_utils.py](raptor/cluster_utils.py))
- UMAP has no `random_state`, so builds aren't reproducible.
- The local-to-global index mapping uses an O(n²·d) equality match, which breaks on duplicate chunks. Track indices directly instead.
- `RAPTOR_Clustering.perform_clustering` has no `self` or `@staticmethod`, and its recursive call drops `reduction_dimension`, `threshold` and `tokenizer`.
- `ClusterTreeConfig` uses a mutable default `clustering_params={}`.

**`ClusterTreeBuilder`** ([cluster_tree_builder.py](raptor/cluster_tree_builder.py)) changes `self.num_layers` on early stop. That value leaks into later builds, and `Tree.num_layers` can go stale.

**Retrieval** ([tree_retriever.py](raptor/tree_retriever.py), [tree_builder.py](raptor/tree_builder.py))
- `threshold` selection keeps nodes with distance `> threshold`, which is inverted.
- Per-node scipy distance loop: vectorize it, and cache the stacked embedding matrix per tree.
- `RetrievalAugmentation.retrieve` defaults `return_layer_information=True`, but the retriever defaults it to `False`. Make them consistent.

**Config:** `TreeRetrieverConfig` and `TreeBuilderConfig` build OpenAI clients as defaults. Make models required, or default to the configured provider lazily.

**Library hygiene:** remove `logging.basicConfig` from every module and use module-level loggers.

**Concurrency:** the threaded leaf build has no concurrency limit. Use bounded concurrency plus batched embedding calls (one Gemini call per ~100 chunks instead of one per chunk).

### 1c. Packaging and tooling
- `pyproject.toml` (hatchling), Python ≥3.10.
- Optional extras: `gemini`, `local`, `ollama`, `app`, `code`, `mcp`, `dev`.
- Lift the `numpy==1.26.3` pin so NumPy 2 works, which also removes the `pyarrow<20` pin. Drop `openai`, `transformers` and `httpx`.
- ruff (lint + format), mypy (basic), pre-commit, and GitHub Actions CI running pytest on ubuntu and windows.

### 1d. Tests (`tests/`)
- `conftest.py` with deterministic fakes: a hash-based embedder and a summarizer that echoes its first sentence. The unit suite makes no network calls.
- Unit tests for chunking (punctuation preserved, overlap, oversized sentences), clustering (seeded determinism, duplicates, small-n), builder (layer counts, parent/child consistency), retrieval (collapsed, traversal, token budget, threshold direction), storage (round-trip, legacy pickle import of `demo/*` trees), and config (YAML round-trip).
- `@pytest.mark.live` smoke test: build and query the Cinderella sample with Gemini. Skipped without a key.

### 1e. Demo data and docs
- Convert the `demo/*` pickles to the new storage format.
- In the interim Streamlit app, apply only minimal fixes: the `file:///c:/Users/RAGHAV JHA/...` link, and labelling the hardcoded ❌/✅ thematic results as "precomputed". The real UI work is Phase 4.
- Rewrite `README.md` (install, quickstart, CLI, architecture). Add `docs/ARCHITECTURE.md`.

## Phase 2: Experiment harness and strategy sweeps
### 2a. Harness (`raptor/eval/`)
- `datasets/quality.py`: loader plus a seeded, fixed eval split. **Dev subset:** 8 stories (about 70 questions). **Held-out confirm set:** 8 different stories.
- `runner.py`: `raptor eval run configs/exp/*.yaml`. Each config hashes to a run ID, and results go to `artifacts/runs/<id>/` (per-question JSONL plus tree stats).
- Everything goes through the disk cache, so a retrieval-only change reuses cached trees, summaries and embeddings. Most sweeps then cost only one QA call per question.
- Answer extraction: Gemini structured output returning `{"option": int, "rationale": str}`, replacing free-text "Option X" parsing.
- **Metrics:**
  - accuracy, plus accuracy on the QuALITY "hard" subset
  - paired McNemar test and bootstrap CI against the baseline
  - share of retrieved context drawn from summary nodes vs leaf nodes
  - tree stats: depth, branching, summary compression ratio
  - cost: LLM and embedding calls, tokens
  - build and query latency
- `report.py`: auto-generated Markdown with plots, replacing the hand-written `QUALITY_BENCHMARK_REPORT.md`. A dashboard tab browses the runs.
- **Noise caveat:** with ~70 questions, only differences of roughly 10 points or more are reliable. That's why paired tests and a confirm set are used.

### 2b. Strategies to add (each is one registry entry plus a config)
**Chunking**
- sentence-greedy (fixed baseline)
- fixed-token windows with overlap
- paragraph-based
- recursive splitter
- semantic chunking (split where neighbouring-sentence embedding similarity drops)
- chunk sizes 100 / 250 / 500

**Clustering**
- UMAP+GMM (baseline)
- GMM with no reduction, and PCA+GMM
- agglomerative (cosine distance threshold)
- HDBSCAN
- k-means with silhouette-chosen k
- contiguous segmentation: clusters of adjacent chunks, which preserves narrative order (a likely strong option for stories)
- soft vs hard assignment, and the GMM threshold value

**Tree shape and summaries**
- summary length, prompt variants (factual vs narrative vs entity-focused), max layers, stop criterion

**Retrieval**
- collapsed vs traversal
- top-k vs token budget
- hybrid BM25 + dense, fused with RRF
- local cross-encoder reranking (`bge-reranker-base` on the GTX 1650)
- MMR for diversity
- Gemini query vs document `task_type`

**Embeddings**
- `gemini-embedding-001` and `gemini-embedding-2`
- local `bge-small` and `e5-base`, so chunking and clustering sweeps cost no API quota

**Reader** stays fixed at `gemini-3.5-flash-lite` so differences come from retrieval.

### 2c. Sweep protocol (fits the free tier)
- **Stage A:** change one factor at a time from the baseline, on the dev subset.
- **Stage B:** combine the top 2–3 settings per factor.
- **Stage C:** rerun the best 2–3 configs and the baseline on the held-out stories, then publish the report.
- The rate limiter in `settings.py` keeps runs under free-tier RPM/RPD limits, and interrupted runs resume from the cache.

## Phase 3: RAPTOR for codebases (`raptor/code/`)
### 3a. Ingest and parse
- Walk the repo, respecting `.gitignore` and skipping binary, vendored and generated files.
- Parse with tree-sitter (`tree-sitter-language-pack`) using a per-language symbol extractor; Python first.
- Leaf chunks are **symbols**: functions, methods, and classes (header plus docstring). Large symbols are split; tiny ones are merged.
- Each leaf carries metadata: path, line span, signature, docstring, decorators.
- **Contextual chunk text:** prepend a header (`path › Class › def sig`, plus the relevant imports) before embedding.

### 3b. Hierarchy (a DAG, not a strict tree)
- **Structural layers:** symbol → class → file → package/directory → repo. Summaries are built bottom-up with code-specific prompts covering purpose, public API, inputs and outputs, side effects, and dependencies.
- **Semantic layer:** cluster file and class summaries with the best clustering method from Phase 2. This produces cross-cutting "concept" nodes (e.g. "authentication flow", "persistence layer") that link files from different directories. A node can have several parents, so the `types.py` DAG is required.
- **Graph edges** stored alongside the tree: imports, calls (best-effort name resolution), and inheritance.

### 3c. Retrieval for code
- Hybrid lexical + dense:
  - BM25 over identifiers split on camelCase and snake_case, plus docstrings
  - dense search using Gemini `CODE_RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT` task types
- **Graph expansion:** add the callers, callees and parent class of each hit, within the token budget.
- **Level routing:** "how does X work / where is Y handled" questions favour summary nodes; exact symbol questions favour leaves.
- Answers cite `path:line` from node metadata.

### 3d. Incremental indexing
- Store a content hash per file and per symbol. `raptor index --update` uses git diff and mtimes to re-parse changed files, re-embed changed symbols, and re-summarize only their ancestors. The summary cache is keyed by content hash, so unchanged code is never re-summarized.
- For the free tier: during development, summaries can run on the local Ollama model. A `--summary-depth file` option skips per-function summaries.

### 3e. Interfaces
- **CLI:** `raptor index <path>`, `raptor ask "<q>" [--repo]`, `raptor search`, `raptor tree` (print the hierarchy).
- **MCP server** (`raptor serve-mcp`, Python `mcp` SDK) with tools: `search_code`, `ask_codebase`, `get_symbol`, `explain_module`, `get_callers`, `get_callees`. This lets Claude Code and IDEs query an index.

### 3f. Evaluation for code
- **Localization (objective, cheap):** use SWE-bench Lite issues from small repos (`flask`, `requests`, `seaborn`; about 40 issues) and measure whether the gold edited files and functions appear in the top-k results (recall@k, MRR). Retrieval only, with no reader calls.
- **Repo QA:** about 40 hand-verified architecture questions ("where / how / why") on 2–3 repos, including this one. Graded with a rubric by an LLM judge, with human spot checks.
- **Baselines:**
  - flat line-window RAG
  - flat symbol-chunk RAG
  - structural tree
  - structural + semantic tree
  - each of the above with and without graph expansion and hybrid lexical search

## Phase 4: React UI (`web/`) and API (`raptor/server/`)
### 4a. Backend: FastAPI
- **Indexes:** `GET/POST /indexes`, for documents (QuALITY stories, uploaded text/PDF) or code repos.
- **Builds:** run as background jobs. `GET /jobs/{id}/events` streams progress over SSE: chunking → embedding → clustering → summarizing layer N, with counts and cost.
- **Query:**
  - `POST /query` runs retrieve + answer and streams the answer token by token over SSE. Returns the selected node IDs, their layers and scores.
  - `POST /compare` runs Flat RAG vs RAPTOR side by side with the same budget.
- **Tree data:**
  - `GET /indexes/{id}/tree` returns nodes and edges, paged by layer.
  - `GET /indexes/{id}/projection` returns 2-D UMAP coordinates for every node, plus cluster IDs.
- **Experiments:** `GET /runs`, `/runs/{id}`, `/runs/compare?ids=` (metrics, CIs, per-question diffs).
- **Code:** `GET /code/{id}/files`, `/code/{id}/symbol/{sid}`, `/code/{id}/graph`, `POST /code/{id}/ask` (answers with `path:line` citations).
- **Contract and tests:** an OpenAPI schema generates TypeScript types for the frontend (`openapi-typescript`). Backend tests use pytest + `TestClient` with the fake models.
- `raptor serve` serves the API and the built frontend from one process.

### 4b. Frontend: Vite + React + TypeScript
- **Stack:**
  - Tailwind with shadcn/ui components
  - TanStack Query and React Router
  - React Flow for tree/graph views
  - visx or deck.gl for the embedding scatter
  - Shiki for code display
  - Framer Motion for transitions
- Dark and light themes.

**Pages:**
1. **Library:** index cards (documents and repos) with build status. A "New index" flow takes text, a file upload, a QuALITY story or a repo path, plus a strategy preset.
2. **Build view:** live pipeline progress from SSE. Layers appear one by one as clusters form and summaries arrive.
3. **Tree Explorer:**
   - A zoomable layered DAG of the index. Click a node to see its summary, children, source span and cluster.
   - A second view, the embedding map: a 2-D scatter coloured by cluster, with summary nodes drawn as hulls around their children.
4. **Ask / Arena:**
   - Ask a question and watch the answer stream in. The retrieved nodes light up in the tree, and their layer mix shows as a bar.
   - Arena mode runs Flat vs RAPTOR side by side, with the QuALITY gold answer and options when the question comes from the benchmark.
5. **Experiments:** run table, accuracy charts with CI bars, the effect of each strategy factor, the McNemar result vs the baseline, and a question-level diff ("which questions did config B fix or break").
6. **Code Explorer:** repo file tree, a symbol panel, a call/import graph, and Ask with clickable `path:line` citations that open the code viewer at the right lines.

**Frontend tests:** Vitest for components, plus one Playwright smoke run (build an index with fake models → ask → tree highlights).

---

## Order and parallelism
Phase 0 → Phase 1 must come first, since everything builds on the new interfaces, storage and tests. After Phase 1 merges, three tracks can run in parallel sessions because they touch separate directories:
- Phase 2: `raptor/eval/`, `configs/`
- Phase 3: `raptor/code/`
- Phase 4: `raptor/server/`, `web/`

The UI's core pages (Library, Build, Tree Explorer, Ask/Arena) need only Phase 1. The Experiments and Code pages land as Phases 2 and 3 deliver their APIs. Phase 3 picks up Phase 2's winning clustering method once it's known; until then it uses the default.

## Verification
- **Phase 1:**
  - `pytest` (unit suite, offline) green on Windows locally and in CI; `ruff check` and `mypy` clean.
  - `pytest -m live` builds and queries Cinderella with Gemini.
  - The converted `demo/*` trees load and give the same retrieval node IDs as the old pickles for a fixed set of queries.
  - `pip install -e .[gemini,app]` in a fresh venv works.
- **Phase 2:**
  - A baseline config reproduces the current QuALITY numbers within noise.
  - Rerunning a config is fully cached (zero API calls) and gives identical results with seeded UMAP.
  - The report is generated from the run artifacts.
- **Phase 3:**
  - `raptor index` on this repo and on `requests`; `raptor ask` returns answers that cite `path:line`.
  - After editing one function, `raptor index --update` re-summarizes only that function's ancestors (asserted from cache-hit counts).
  - The MCP server is registered in Claude Code, and its tools return results.
  - The localization benchmark produces recall@k for every baseline.
- **Phase 4:**
  - Backend: `pytest` API tests pass, and the generated TS types compile (`tsc --noEmit`).
  - Frontend: `vitest` passes and the Playwright smoke run is green.
  - Manual check in the browser pane via `raptor serve`, covering the whole flow: build a QuALITY story with live progress, explore the tree and the embedding map, run an Arena comparison with streaming answers, open an experiment comparison, ask a code question and follow a citation. Check both themes and a phone-width viewport.
  - Retire Streamlit `app.py` once everything above works.
