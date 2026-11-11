# Architecture

```
raptor/
  config.py        RaptorConfig (pydantic, YAML round-trip, fingerprint for run IDs)
  settings.py      env/.env settings: API key, cache dir, rate limits
  types.py         Node / Tree (layered DAG; cached normalized embedding matrix)
  registry.py      name -> factory registry for pluggable strategies
  chunking/        Chunker ABC + "sentence" strategy        (CHUNKERS registry)
  clustering/      ClusteringAlgorithm ABC + "raptor_gmm"    (CLUSTERERS registry)
  builder.py       TreeBuilder: chunk -> embed -> (cluster -> summarize -> embed)*
  retrieval/       TreeRetriever (collapsed, traversal), FlatRAG baseline
  models/          base interfaces, Gemini (google-genai), local (sentence-transformers),
                   Ollama, SQLite disk cache, shared rate limiter
  storage.py       index format v1 (tree.json + embeddings.npz) and legacy pickle import
  pipeline.py      RetrievalAugmentation (+ backward-compatible RetrievalAugmentationConfig)
  cli.py           `raptor build | ask | info | migrate`
  eval/datasets/   QuALITY loader
```

## Data flow

1. **Chunking.** A `Chunker` turns text into `Chunk`s, which carry character offsets.
2. **Leaf embedding.** All chunks are embedded in one batched call, which Gemini splits into requests of up to 100 texts.
3. **Layers.** For each layer:
   - Stop when the layer has `stop_threshold` nodes or fewer (default `reduction_dimension + 1`), or when clustering stops reducing the node count.
   - Otherwise, the clusterer returns soft clusters of row indices, each cluster is summarized (with bounded concurrency), and the summaries are embedded in one batch.
4. **Retrieval.**
   - *Collapsed* mode scores every node with one matrix product and fills `top_k` within a token budget.
   - *Traversal* mode starts at a layer and descends into the children of the best nodes.

`TreeBuilder` emits progress events (`chunking`, `embedding`, `clustering`, `summarizing`, `done`). The CLI prints them, and the web UI will stream them.

## Adding a strategy

```python
from raptor.clustering import CLUSTERERS, ClusteringAlgorithm


class Agglomerative(ClusteringAlgorithm):
    def cluster(self, embeddings, token_counts):
        # Return list[list[int]]; every row must be in at least one cluster.
        raise NotImplementedError


@CLUSTERERS.register("agglomerative")
def _make(**params):
    return Agglomerative(**params)
```

Then select it with `clustering: {strategy: agglomerative}` in a config.

## Caching

`models.build_models()` wraps the models in `Cached*Model`, which stores results in `.cache/raptor/cache.sqlite`. Each cache key combines the model id, the call parameters and the input text. As a result, rebuilding the same tree or rerunning the same evaluation makes no API calls. The embedding wrapper deduplicates repeated texts within a batch. Each wrapper's `stats` counts hits and misses.

## Index format

An index is a directory holding two files:
- `tree.json` stores nodes (text, children, layer, metadata), the layer lists, and tree metadata (embedding model id, config snapshot).
- `embeddings.npz` stores one float32 matrix per embedding key.

Saves are atomic. Loading never executes code. `raptor migrate` converts the original pickles using an allow-listed unpickler.

## Changes from the original implementation

| Area | Original | Now |
|---|---|---|
| Chunking | regex split on `.!?` deleted sentence punctuation; overlap mixed indices | punctuation and offsets preserved; overlap is a real token budget; no chunk exceeds `max_tokens` |
| Clustering | unseeded UMAP; O(n²·d) index recovery that mis-mapped duplicate chunks; recursion dropped params and could loop forever; GMM crashed on collapsed points | seeded; direct index tracking; params kept; guaranteed termination; degenerate GMM fits skipped |
| Builder | early stop mutated `num_layers` for later builds; unbounded threads; one embedding call per chunk | per-build layer count; bounded workers; batched embeddings; progress events |
| Retrieval | `threshold` mode kept the *least* similar nodes; Python loop over scipy distances | similarity threshold; one vectorized matrix product with cached matrix |
| Pipeline | `add_documents` called `input()`; `add_to_existing` missing; OpenAI clients created as defaults | explicit `overwrite=`; `add_to_existing` reuses leaf embeddings; models built lazily from config |
| Storage | `pickle` (runs arbitrary code on load) | JSON + npz; restricted legacy import |
| Gemini | deprecated `google-generativeai`; hand-written retry; no task types | `google-genai`; tenacity retries on 429/5xx; shared RPM limiter; retrieval task types |
| Secrets | API key hardcoded in a script | `.env` / environment via pydantic-settings |
| Logging | `logging.basicConfig` in every module | module loggers; only the CLI configures logging |
