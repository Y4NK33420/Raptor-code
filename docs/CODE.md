# RAPTOR for codebases (`raptor.code`)

`raptor.code` builds a RAPTOR-style hierarchy over a source repository and answers questions about it with `path:line` citations. It reuses the Phase 1 `Node`/`Tree` types, model wrappers (key pool, disk cache) and storage.

Install: `pip install -e ".[code,gemini,mcp]"` (add `local` for offline sentence-transformers embeddings).

## Usage

```bash
raptor code index path/to/repo --out indexes/repo          # build (summary depth "file" by default)
raptor code index . --out indexes/me -x "tests/fixtures/" # extra gitignore-style excludes
raptor code index repo --out idx --summary-depth symbol   # also summarize every symbol
raptor code index repo --out idx -c my.yaml               # YAML CodeConfig (models, chunking, retrieval)
raptor code search idx "retry with backoff" -k 10 --bm25 --graph
raptor code ask idx "How are redirects handled?"          # answer + path:line citations
raptor code tree idx --depth 3                            # print the hierarchy
raptor code update idx                                    # re-index changed files only
raptor code serve-mcp idx                                 # MCP server over stdio
```

Python API:

```python
from raptor.code import CodeIndex
index = CodeIndex.build("path/to/repo")
index.save("indexes/repo")
index = CodeIndex.load("indexes/repo")
hits = index.search("session cookies", k=10)       # list[CodeHit]
answer = index.ask("How are cookies persisted?")   # CodeAnswer(answer, citations)
index.symbol(hits[0].node_index)                   # hit, children, parents, callers, callees
index.graph(("calls", "imports", "inherits"))      # list[CodeEdge]
index.update()                                     # {changed_files, reembedded, resummarized}
```

Exports: `CodeIndex`, `CodeConfig`, `CodeHit`, `CodeAnswer`, `CodeEdge`.

## Architecture

1. **Ingest** (`ingest.py`): walks the repo (`git ls-files` when available, else the file system) honouring `.gitignore`, skipping binary, vendored, generated and oversized files.
2. **Parse** (`languages/`): tree-sitter (`tree-sitter-language-pack`) extractor per language (Python so far) yields symbols (functions, methods, classes), signatures, docstrings, decorators, imports and call sites.
3. **Leaf chunks** (`chunks.py`): one leaf per symbol; large symbols are split, tiny neighbours merged. Every chunk is prefixed with a contextual header (`path > Class > def signature`, plus relevant imports) before embedding.
4. **Hierarchy** (`build.py`): symbol -> class -> file -> directory -> repo, summarized bottom-up with code-specific prompts (purpose, public API, I/O, side effects, dependencies). A semantic layer clusters file/class summaries into cross-cutting *concept* nodes; a node may have several parents (DAG).
5. **Graph** (`graph.py`): imports, calls (best-effort name resolution) and inheritance edges stored in `graph.json`.
6. **Search** (`search.py`, `lexical.py`): BM25 over identifiers split on camelCase/snake_case plus dense retrieval, fused with reciprocal-rank fusion; level routing (broad "how/where" questions favour summaries, identifier-like queries favour leaves); descent from summary hits into their best leaves; call-graph expansion with a decay factor.
7. **Answers** (`index.py`): retrieved nodes are packed into a token budget, each tagged `[path:start-end]`; the reader is told to cite those locations.
8. **Incremental update**: content hashes per file and per symbol. `update()` re-parses changed files (git diff or hashes), re-embeds changed symbols and re-summarizes only their ancestors; summaries are additionally cached by content hash so unchanged text is never re-summarized (asserted in `tests/test_code.py::test_update_resummarizes_only_ancestors`).

On-disk layout: `code.json` (config, file table, hashes), `graph.json`, `tree/` (Phase 1 tree format).

## MCP server

Tools: `search_code`, `ask_codebase`, `get_symbol`, `explain_module`, `get_callers`, `get_callees`.

Register with Claude Code (user scope, absolute paths; the server needs `GEMINI_API_KEY`(S) in its environment or in the `.env` next to where it is launched, unless the index uses local models):

```bash
claude mcp add raptor-code -- /path/to/.venv/bin/raptor code serve-mcp /path/to/indexes/repo
# Windows: claude mcp add raptor-code -- D:\path\.venv\Scripts\raptor.exe code serve-mcp D:\path\indexes\repo
```

`scripts/mcp_check.py <index>` exercises all six tools through the MCP SDK's stdio client (no Claude Code config involved).

## Notes on free-tier use

- `--summary-depth file` (default) skips per-function summaries (one call per file, plus classes/dirs/concepts).
- All model calls go through the disk cache; rebuilding or re-running evaluations is free for unchanged content.
- The embedding provider is configurable (`models.embedding_provider: local`). A query must be embedded with the same model as the index, so an index built with local embeddings needs no Gemini quota to search.
