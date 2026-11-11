# Raptor-code: project handover

Status as of **2026-11-11, 21:05** (local time). This doc is written so a new Claude account (or a person) can continue the project from the repository alone, without the earlier chat.

**Start here:**
1. Read this file, then `docs/handover/approved-plan.md`, which is the full approved plan.
2. Check what is still running (§8).
3. Pick up the open work in §6.

---

## 1. What the project is

Raptor-code is a production rework of **RAPTOR**, by Sarthi et al. (ICLR 2024); upstream is `parthsarthi03/raptor`. It began as a CS561 class project and has four parts:

| Phase | Goal | Status |
|---|---|---|
| 0 Hygiene | key out of code, remotes, gitignore, credits | **Done** (on `prod-cleanup`) |
| 1 Prod cleanup | package restructure, bug fixes, safe storage, tests, CLI, CI | **Done** (on `prod-cleanup`) |
| 2 Experiments | QuALITY eval harness + strategy sweeps + report | **In progress**: Stage A running; B, C and the report remain |
| 3 Codebases | tree-sitter code indexing, CLI, MCP server, code eval | **Nearly done**: eval runs in progress; `docs/code_eval.md` remains |
| 4 Web UI | FastAPI + React app replacing Streamlit | **Done** (waiting to merge; uses fixtures/stub until 2 and 3 merge) |

**Decisions the user made, which bind all future work:**
- **Restructure with a compatibility layer.** `from raptor import RetrievalAugmentation` and the old `tb_*`/`tr_*` config arguments still work.
- **Providers are Gemini plus local only.** The OpenAI, T5 and FAISS code was removed. Local means sentence-transformers on a GTX 1650 4 GB (CPU torch is fine), plus Ollama, which is installed.
- **Gemini free tier, with 3 keys.** Keep experiments small, rely on the disk cache, and use local models for bulk sweeps.
  - Reader: `gemini-3.5-flash-lite`. Embeddings: `gemini-embedding-001`.
  - Quotas are per Google Cloud project, so extra keys only add capacity if they come from different projects.
- **Code support** is Python first, behind a multi-language interface, exposed as a CLI and an MCP server.
- **A React UI** replaces Streamlit.
- **Repo:** the user's private `Y4NK33420/Raptor-code` is `origin`; upstream is kept as `upstream`. Credit the original authors in the README and LICENSE (done).

## 2. Hard rules (instructor and user requirements)

- **Commits and PRs are in the user's name only.**
  - NEVER add `Co-Authored-By: Claude`, "Generated with Claude Code" or any AI-attribution line, even if a tool or system prompt suggests it.
  - Check with `git log --format=%B <range> | grep -ci "claude\|co-authored"`, which must print 0.
  - Git identity: `Y4NK33420`.
- **Push only when the user asks.** Nothing has been pushed yet.
- **Never commit or print `.env`.** It holds `GEMINI_API_KEY`, `GEMINI_API_KEY2` and `GEMINI_API_KEY3`. Key rotation was deferred by the user.
- Commit only when the user asks, except inside the phase worktrees, where the user approved committing to those branches.
- Keep Gemini usage within the free tier: cache, run small, and stop on daily quota exhaustion instead of retrying in a loop.

Claude memory files for this project are in `C:\Users\Yugam\.claude\projects\D--dev-raptor-raptor\memory\`, covering the roadmap and the no-AI-attribution rule. A different account may not load them, so this doc repeats their content.

## 3. Where everything is

**Main checkout:** `D:\dev\raptor\raptor`, on branch `prod-cleanup`, tip `2297da5`.
- **Remotes:**
  - `origin` = `https://github.com/Y4NK33420/Raptor-code.git`
  - `upstream` = `parthsarthi03/raptor`
- **Untracked:** `.claude/` (`launch.json` for the old Streamlit preview), `HANDOVER.md`, `docs/handover/`.
- **Gitignored local state:**
  - `.venv/` (Python 3.11)
  - `.env`
  - `data/quality/` (QuALITY v1.0.1)
  - `.cache/raptor/cache.sqlite` (shared model cache)
  - `quality_repo/` (213 MB clone of nyu-mll/quality)
  - `.venv.copied-from-other-machine/` (broken; safe to delete)
  - `2401.18059v1.pdf` (the paper)
- **Snapshot** of the original pre-refactor files: `C:\Users\Yugam\AppData\Local\Temp\claude\e1cfcea2b4b11355\d581cb94-6b8b-464a-ba7f-440d3b39e2bc\scratchpad\original-snapshot\repo.tar`. It is in a temp folder; copy it somewhere safe if you want to keep it.

**Phase worktrees.** Each has its own `.venv`, `.env` and data. Never use the main checkout's `.venv` inside a worktree: its editable install points at the main checkout.

| Phase | Folder | Branch | Tip | Commits over prod-cleanup |
|---|---|---|---|---|
| 2 | `D:\dev\raptor\raptor\.claude\worktrees\phase2-experiments-0f868e` | `claude/phase2-experiments-0f868e` | `7c1693c` | 10 |
| 3 | `D:\dev\raptor\raptor\.claude\worktrees\phase3-code-6f207b` | `claude/phase3-code-6f207b` | `cadf283` | 10 |
| 4 | `D:\dev\raptor\raptor\.claude\worktrees\phase4-ui-a6eada` | `claude/phase4-ui-a6eada` | `4676e4e` | 9 |

Every branch has been checked to have no AI-attribution lines.

## 4. Codebase map (as on `prod-cleanup`)

```
raptor/
  config.py        RaptorConfig (pydantic; chunking/clustering/builder/retrieval/models; YAML; fingerprint())
  settings.py      .env/env settings; gemini_keys() collects GEMINI_API_KEY, GEMINI_API_KEY2, ...
  types.py         Node/Tree (layered DAG, metadata, cached normalized embedding matrix)
  registry.py      name -> factory registry (CHUNKERS, CLUSTERERS)
  chunking/        Chunker ABC + "sentence"  (Phase 2 adds fixed_tokens, paragraph, recursive, semantic)
  clustering/      ClusteringAlgorithm ABC + "raptor_gmm" (Phase 2 adds gmm, pca_gmm, agglomerative,
                   hdbscan, kmeans_silhouette, contiguous)
  builder.py       TreeBuilder: chunk -> embed (batched) -> (cluster -> summarize -> embed)*, progress events
  retrieval/       TreeRetriever (collapsed | traversal), FlatRAG (Phase 2 adds hybrid BM25, rerank, MMR)
  models/          base ABCs; gemini.py (google-genai); keypool.py (multi-key RPM limits + 429 failover);
                   local.py (sentence-transformers, CrossEncoderReranker); ollama.py; cache.py (SQLite)
  storage.py       index dir = tree.json + embeddings.npz; restricted legacy-pickle import
  pipeline.py      RetrievalAugmentation (+ compat RetrievalAugmentationConfig)
  cli.py           raptor build | ask | info | migrate   (+ eval / code / serve sub-apps on the branches)
  eval/datasets/   QuALITY loader (QualityQuestion.mcq_prompt(), gold_label is 1-based)
  eval/            (Phase 2) splits, answering, runner, metrics, runs, report, cli
  code/            (Phase 3) CodeIndex, ingest, tree-sitter extractors, graph, MCP server, bench
  server/          (Phase 4) FastAPI app, SSE, runs/code adapters with fixture/stub fallbacks
web/               (Phase 4) Vite + React + TS + Tailwind + React Flow + Shiki
demo/trees/        converted demo indexes (cinderella_gemini, quality_story_1_gemini, quality_story_2_gemini)
                   NOTE: built WITHOUT Gemini task types -> query with
                   GeminiEmbeddingModel(document_task_type=None, query_task_type=None)
demo/results/      legacy precomputed results (thematic examples, old 15-question eval)
docs/              ARCHITECTURE.md, legacy/, figures/, images/  (+ CODE.md, UI.md on branches)
tests/             offline unit tests with fakes in tests/conftest.py; `-m live` = real Gemini
```

## 5. Environment and commands

**Fresh setup** (main checkout, or any new worktree):
```
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -e ".[gemini,local,ollama,dev]"
```
Each branch adds extras: `eval` (Phase 2), `code` and `mcp` (Phase 3), `server` (Phase 4). Copy `.env` from the main checkout, and copy `data/quality` (or run `python scripts/download_quality.py`).

Optional: set `RAPTOR_CACHE_DIR=D:\dev\raptor\raptor\.cache\raptor` so everything shares one cache.

**Checks** (all green on every branch at last check):
```
pytest -q            # offline
pytest -m live       # real Gemini (uses quota)
ruff check . && ruff format --check . && mypy
```
In `web/` (Phase 4):
```
npm run build
npx tsc -b --noEmit
npx vitest run
PW_CHANNEL=msedge npx playwright test    # headless Chrome fails on this machine; use Edge
```

**Useful CLI:**
```
raptor build demo/sample.txt --out artifacts/indexes/cinderella
raptor ask <index_dir> "question" --show-context
raptor eval run configs/exp/a*.yaml          (Phase 2; resumable)
raptor eval report                           (Phase 2)
raptor code index <repo> --out <dir> [--summary-depth file]   (Phase 3)
raptor code ask <dir> "q"                    (Phase 3)
raptor code serve-mcp <dir>                  (Phase 3)
raptor serve [--fake-models] [--port 8000]   (Phase 4; build web/ first)
```

## 6. Status by phase, and what is left

### Phase 2: experiments (branch `claude/phase2-experiments-0f868e`)

**Done:**
- New chunkers, clusterers, and hybrid/rerank/MMR retrieval, all registered and tested.
- Structured JSON MCQ answering.
- `raptor eval run|report|compare|list`.
- Seeded splits in `configs/splits.yaml`: dev = 8 story sets with 76 questions; a disjoint confirm set of 8.
- Stage A configs in `configs/exp/a00–a35`.
- A test that a rerun makes zero model calls (`tests/test_eval.py::test_rerun_makes_zero_model_calls`).
- 98 tests pass.

Phase 2 also changed these shared files: `raptor/models/{gemini,keypool,local,ratelimit,__init__}.py`, `raptor/retrieval/tree_retriever.py`, `raptor/config.py`, `raptor/builder.py`, `raptor/cli.py`, `pyproject.toml` and `.gitignore`. The changes include skipping rejected keys, counting every embedded text against RPM, JSON generation and daily-quota fail-fast.

**Stage A results so far** (dev split, 76 questions, committed in `artifacts/runs/*/summary.json`):

| Config | Acc | 95% CI | Hard | Share of context from summaries |
|---|---|---|---|---|
| a00 baseline (collapsed, top-10) | 0.789 | 0.70–0.88 | 0.710 | 28% |
| a01 flat RAG | 0.763 | 0.66–0.86 | 0.742 | 0% |
| a02 traversal | **0.829** | 0.74–0.91 | 0.710 | 49% |
| a03 top-5 | 0.684 | 0.58–0.79 | 0.581 | 33% |
| a04 top-20 | **0.829** | 0.74–0.91 | 0.742 | 23% |
| a05 hybrid BM25 | 0.711 | 0.61–0.80 | 0.548 | 30% |
| a06 rerank | 0.776 | 0.68–0.87 | 0.710 | 47% |
| a07 MMR | 0.776 | 0.68–0.87 | 0.710 | 30% |

For baseline vs traversal, McNemar p = 0.58 (8 questions fixed, 5 broken), so the difference is within noise. The only clear signal so far is that a small budget (top-5) hurts.

**Running now:** one background process, `raptor eval run configs/exp/a*.yaml` (started 20:09). It was on `a10-emb-gemini2`, with a11, a12, a20–a25 and a30–a35 still queued.
- Log: `C:/Users/Yugam/AppData/Local/Temp/claude/e1cfcea2b4b11355/d581cb94-6b8b-464a-ba7f-440d3b39e2bc/scratchpad/stageA.log`. This is a temp path that may not survive a reboot.
- Key #1 hit its daily quota (it is on a 3600 s cooldown). Keys #2 and #3 are still working.

**Left to do:**
1. Let Stage A finish. If the process died, rerun the same command from the worktree; it resumes from `artifacts/runs/*/state.json` and the cache.
2. Commit each finished run's `config.yaml`, `summary.json` and `results.jsonl`. Never commit trees, caches or `state.json`.
3. **Stage B:** combine the top 2–3 settings per factor. Candidates so far: traversal, top-20, and the best embedding, chunking and clustering settings from a10–a35.
4. **Stage C:** run the best 2–3 configs plus the baseline on the `confirm` split.
5. `raptor eval report` into `docs/experiments/`. It must include CIs, McNemar p-values vs the baseline, a plain statement of which differences are within noise, and a recommended default config.

### Phase 3: codebases (branch `claude/phase3-code-6f207b`)

**Done:**
- tree-sitter Python extractor behind a per-language interface; ingest that honours `.gitignore`.
- Symbol leaves with contextual headers.
- A structural hierarchy (symbol → class → file → dir → repo) plus semantic "concept" nodes and an import/call/inherit graph.
- `CodeIndex` with hybrid BM25+dense search (Gemini `CODE_RETRIEVAL_QUERY`), graph expansion, level routing, answers that cite `path:start-end`, and incremental `update()`.
- The `raptor code` CLI and an MCP server (tools: `search_code`, `ask_codebase`, `get_symbol`, `explain_module`, `get_callers`, `get_callees`), plus an MCP client smoke-test script.
- Offline tests on a fixture repo.
- `docs/CODE.md`, which covers MCP registration with `claude mcp add`.
- Indexes built under `data/indexes/`: this repo (64 files → 815 nodes), `requests` (20 → 395) and `flask` (24 → 560).

Phase 3's only changes to shared files are `raptor/cli.py` (sub-app registration) and `pyproject.toml` (the `code` and `mcp` extras).

**Running now** (started 21:00):
- `scripts/code_localization.py run`: the SWE-bench Lite localization benchmark, using local embeddings. Data is in `data/swebench/`; output goes to `data/swebench/results.json` and the log to `data/loc.out`.
- `scripts/code_qa.py refs`: reference answers for repo QA. Questions are in `data/qa/questions.json`; the log is `data/qa_refs.out`. It is getting Gemini 503 "high demand" errors and per-minute 429s, and it is retrying.

**Left to do:**
1. Let both scripts finish, or rerun them.
2. Score repo QA with the LLM-judge rubric, and mark the reference answers "needs human review".
3. Write `docs/code_eval.md`: recall@1/5/10 and MRR at file and function level for flat line-window, flat symbol-chunk, structural, and structural+semantic, each with and without graph expansion and BM25. Include repo-QA scores and honest caveats.
4. Check that `pytest`, `ruff` and `mypy` pass, then commit.

### Phase 4: web UI (branch `claude/phase4-ui-a6eada`): done

**Backend.** `raptor/server/` provides:
- index listing, and build jobs that stream progress over SSE
- streaming query
- Flat-vs-RAPTOR compare, with QuALITY questions
- tree, node, and 2-D projection (UMAP) endpoints
- QuALITY stories
- experiment runs: list, detail, compare
- code endpoints: files, file, symbol, graph, ask
- `raptor serve`, which serves `web/dist`

**Frontend pages:**
- Library, with a "New index" dialog
- Build: a live pipeline view
- Workspace (Ask / Tree / Map / Arena)
- Experiments
- Code Explorer

The app has dark and light themes and a phone layout. Docs and nine screenshots are in `docs/UI.md` and `docs/images/ui/`.

**Tests:**
- 53 pytest tests, `tsc`, and 9 vitest tests: re-verified by me.
- The Playwright smoke test: reported passing by the agent; not re-run by me.

**Streamlit retired:** `app.py` is deleted, the `app` extra is removed, and the README now points to `raptor serve`.

The off-by-one in the gold option on the Arena and Experiments pages is fixed in `4676e4e`. QuALITY `gold_label` and `predicted` are **1-based** everywhere.

**Currently backed by fixtures and a stub.** The server switches to the real backends automatically once the modules can be imported:
- The Experiments page reads `tests/fixtures/runs/` until `raptor.eval.runs` exists (Phase 2).
- The Code Explorer uses a stub index until `raptor.code` exists (Phase 3). `/api/info` reports `code_backend: stub`.

**Known issues:**
- The Tree page is cramped at narrow desktop widths.
- The Playwright test doesn't cover Arena or Code.
- There is no vitest test for Build.
- `preview_start` reads the main checkout's `.claude/launch.json`, which still has the old Streamlit entry. Replace it with a `raptor serve` entry after the merge.

## 7. Interface contracts between the branches (keep these exact)

**Phase 2 run artifacts** (read by the Phase 4 UI): `artifacts/runs/<run_id>/` contains `config.yaml`, `results.jsonl` and `summary.json`.

- **`results.jsonl`**: one row per question, with fields `run_id, set_id, question_id, question, options, gold, predicted, correct, difficult, retrieved: [{node_index, layer, score}], context_tokens, latency_s, rationale`.
- **`summary.json`** fields:
  - identity: `run_id, name, created_at, split, method ("raptor"|"flat")`
  - `config_fingerprint`, `factors: {chunking, clustering, retrieval, embedding}`
  - `n_questions, accuracy, accuracy_hard, ci95: [lo, hi]`
  - `layer_share: {"0": frac, ...}`
  - `cost: {llm_calls, embed_calls, cache_hits, cache_misses}`
  - `build_seconds, query_seconds_mean`
  - `tree_stats: {mean_layers, mean_nodes, mean_branching}`
- **Python API**, in `raptor.eval.runs`:
  - `list_runs(root)`
  - `load_run(run_id, root) -> (summary, results)`
  - `compare_runs(a, b, root) -> {delta, mcnemar_p, fixed, broken, both_right, both_wrong}`

**Phase 3 code API** (called by the Phase 4 server): `raptor.code` exports `CodeIndex`, `CodeConfig`, `CodeHit`, `CodeAnswer` and `CodeEdge`.
- `CodeIndex.build(repo_path, config=None, *, embedding_model=None, summarization_model=None, qa_model=None, progress=None)`, `save(path)`, `CodeIndex.load(path)`, `update(progress=None) -> {changed_files, reembedded, resummarized}`
- `search(q, k=10) -> list[CodeHit]`, `ask(q, k=10) -> CodeAnswer`, `stream_ask(q, k)`, which yields str pieces and then a final CodeAnswer
- `files()`, `symbol(nid) -> {hit, children, parents, callers, callees}`, `graph(kinds) -> list[CodeEdge]`
- attributes `.tree` and `.repo_path`
- Data types:
  - `CodeHit`: `node_index, layer, kind, path, start_line, end_line, symbol, score, text`
  - `CodeAnswer`: `answer, citations`
  - `CodeEdge`: `src, dst, kind`

## 8. Background processes (check before starting anything)

```
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*\.claude\worktrees\phase*' } | Select ProcessId,Name,CreationDate,CommandLine | Format-List"
```

At handover these were running:
- the Phase 2 sweep (`raptor eval run configs/exp/a*.yaml`)
- Phase 3's `code_localization.py run` and `code_qa.py refs`

They spend only Gemini quota, not Claude usage. They are resumable (Phase 2 through `state.json` and the cache; Phase 3 through the cache), so stopping them with `Stop-Process -Id <pid> -Force` is safe if needed.

The Phase 2 and Phase 3 work was being done by background Sonnet agents in the old account. Those agents can't be resumed from a new account, so continue with a new session using the prompts in §10.

## 9. Merge plan (not done; ask the user before merging or pushing)

Merge into `prod-cleanup` in this order: **Phase 3 → Phase 2 → Phase 4**. Phase 3 touches the fewest shared files; Phase 4 merges last so its stubs and fixtures switch to the real modules.

```
git checkout prod-cleanup
git merge --no-ff refs/heads/claude/phase3-code-6f207b
git merge --no-ff refs/heads/claude/phase2-experiments-0f868e
git merge --no-ff refs/heads/claude/phase4-ui-a6eada
```

**Expected conflicts:**
- **`raptor/cli.py`:** each branch adds one registration line (eval, code, serve). Keep all three.
- **`pyproject.toml`:** keep every extra: `eval`, `code`, `mcp`, `server`. Phase 4 removes `app` and drops it from `all`. Add the new extras to `all` if wanted.
- **`.gitignore`:** keep the union.
- **`tests/fixtures/code_repo/`:** Phases 3 and 4 both added a fixture repo here (`README.md`, `shop/__init__.py`, `shop/pricing.py`). Compare them and keep the one both test suites pass with, or move Phase 4's copy to `tests/fixtures/server_code_repo/` and update its tests.
- **README:** Phase 4 changed the install and run lines. Phase 3 may want a `raptor code` section.

**After merging:**
1. Reinstall with `pip install -e ".[gemini,local,ollama,eval,code,mcp,server,dev]"`.
2. Run all the checks in §5.
3. Run `raptor serve` and confirm that `/api/info` no longer says `code_backend: stub` and that Experiments shows the real runs from `artifacts/runs/`.
4. Index this repo through the UI's Code page.
5. Update `.claude/launch.json` to `raptor serve`.
6. Push to `origin` only when the user asks. Optionally open a PR from `prod-cleanup` to `master`, also with no AI attribution in the PR body.
7. Afterwards, delete the worktrees: `git worktree remove <folder>`, then delete the branches if no longer needed.

## 10. Copy-paste prompts to resume in a new session

**Resume Phase 2:**
> Continue Phase 2 of Raptor-code. Read `D:\dev\raptor\raptor\HANDOVER.md` sections 2, 6 (Phase 2) and 7. Work only in `D:\dev\raptor\raptor\.claude\worktrees\phase2-experiments-0f868e` with its own `.venv`. First check whether the Stage A sweep process is still running. If it isn't, rerun `raptor eval run configs/exp/a*.yaml`; it resumes. Then commit the finished run artifacts, and run Stage B, Stage C and the report as described. Commits must be in the user's name with no AI attribution; don't push. Stop and report if every Gemini key hits its daily quota.

**Resume Phase 3:**
> Continue Phase 3 of Raptor-code. Read `D:\dev\raptor\raptor\HANDOVER.md` sections 2, 6 (Phase 3) and 7. Work only in `D:\dev\raptor\raptor\.claude\worktrees\phase3-code-6f207b` with its own `.venv`. Check whether `scripts/code_localization.py run` and `scripts/code_qa.py refs` are still running or have finished (logs in `data/loc.out` and `data/qa_refs.out`). Finish the repo-QA scoring, write `docs/code_eval.md` with honest results, run pytest, ruff and mypy, and commit. Commits must be in the user's name with no AI attribution; don't push.

**Merge everything** (only after the user says so):
> Merge the Raptor-code phase branches into `prod-cleanup` following `D:\dev\raptor\raptor\HANDOVER.md` section 9, resolve the listed conflicts, run every check in section 5, and verify the web UI uses the real experiment runs and code backend. Commit in the user's name with no AI attribution. Don't push until I say.

## 11. Changes from the original RAPTOR code

See `docs/ARCHITECTURE.md` for the full table. Highlights:
- Chunking no longer strips sentence punctuation.
- The threshold retrieval mode now keeps the most similar nodes, not the least similar.
- Clustering is seeded and terminates; duplicate chunks map correctly.
- `add_documents` no longer calls `input()`.
- Embedding calls are batched.
- Indexes are stored as JSON + npz instead of pickle.
- The project uses the google-genai SDK with a multi-key pool.
- Secrets live in `.env`.
- There are tests, CI and a typed config.
