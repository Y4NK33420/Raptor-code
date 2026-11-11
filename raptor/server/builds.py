"""Index builds with fine-grained progress for the live Build view.

The :class:`~raptor.builder.TreeBuilder` already reports stage progress. Here its chunker,
clusterer, summarizer and embedder are wrapped so the UI also learns *which* nodes were
grouped into which cluster and the text of each summary as it arrives, which lets it draw
the tree growing layer by layer.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..builder import TreeBuilder
from ..chunking import Chunk, Chunker, default_tokenizer, make_chunker
from ..clustering import ClusteringAlgorithm, make_clusterer
from ..config import RaptorConfig
from ..models import ModelBundle
from ..models.base import BaseEmbeddingModel, BaseSummarizationModel
from ..storage import save_tree
from .indexes import preview
from .jobs import Job

PRESETS: dict[str, dict[str, Any]] = {
    "balanced": {
        "label": "Balanced",
        "description": "The RAPTOR paper's setup: 100-token chunks, UMAP + GMM soft clustering, "
        "up to five summary layers.",
        "config": {},
    },
    "quick": {
        "label": "Quick",
        "description": "Bigger chunks and at most three layers. About half the summary calls, "
        "which suits the free API tier.",
        "config": {
            "chunking": {"max_tokens": 250},
            "builder": {"max_layers": 3, "summarization_length": 120},
            "clustering": {"reduction_dimension": 8},
        },
    },
    "fine": {
        "label": "Fine-grained",
        "description": "Small chunks and tighter clusters. More nodes and a deeper tree; "
        "best for detail questions.",
        "config": {
            "chunking": {"max_tokens": 60},
            "builder": {"summarization_length": 80},
            "clustering": {"threshold": 0.15},
            "retrieval": {"top_k": 14},
        },
    },
    "shallow": {
        "label": "Single layer",
        "description": "One layer of summaries over the chunks. A cheap middle ground between "
        "flat retrieval and a full tree.",
        "config": {"builder": {"max_layers": 1}, "chunking": {"max_tokens": 150}},
    },
}


def preset_config(name: str, overrides: dict[str, Any] | None = None) -> RaptorConfig:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    data = RaptorConfig().model_dump(mode="json")
    for patch in (PRESETS[name]["config"], overrides or {}):
        for section, values in patch.items():
            if isinstance(values, dict) and isinstance(data.get(section), dict):
                data[section].update(values)
            else:
                data[section] = values
    return RaptorConfig.model_validate(data)


def _context_key(texts: Sequence[str]) -> str:
    # Mirrors raptor.types.nodes_text, which builds the summarizer's input.
    return "".join(" ".join(t.splitlines()) + "\n\n" for t in texts)


class _Tracker:
    """Shared state across the wrappers: node texts and the cluster → parent mapping."""

    def __init__(self, job: Job) -> None:
        self.job = job
        self.lock = threading.Lock()
        self.texts: dict[int, str] = {}
        self.layer_start = 0
        self.layer_count = 0
        self.layer = 0
        self.pending: dict[str, tuple[int, int]] = {}  # context -> (layer, parent id)
        self.summaries_done = 0
        self.llm_calls = 0
        self.embed_calls = 0
        self.embedded_texts = 0
        self.tokenizer = default_tokenizer()

    def cost(self) -> dict[str, int]:
        return {
            "llm_calls": self.llm_calls,
            "embed_calls": self.embed_calls,
            "embedded_texts": self.embedded_texts,
        }


class _TrackingChunker(Chunker):
    def __init__(self, inner: Chunker, tracker: _Tracker) -> None:
        self.inner = inner
        self.tracker = tracker

    def split(self, text: str) -> list[Chunk]:
        chunks = self.inner.split(text)
        t = self.tracker
        with t.lock:
            t.texts = {i: c.text for i, c in enumerate(chunks)}
            t.layer_start, t.layer_count, t.layer = 0, len(chunks), 0
        t.job.emit(
            "chunks",
            {
                "count": len(chunks),
                "chars": len(text),
                "chunks": [
                    {
                        "id": i,
                        "preview": preview(c.text, 90),
                        "tokens": len(t.tokenizer.encode(c.text, disallowed_special=())),
                    }
                    for i, c in enumerate(chunks)
                ],
            },
        )
        return chunks


class _TrackingClusterer(ClusteringAlgorithm):
    def __init__(self, inner: ClusteringAlgorithm, tracker: _Tracker) -> None:
        self.inner = inner
        self.tracker = tracker

    def cluster(self, embeddings: np.ndarray, token_counts: Sequence[int]) -> list[list[int]]:
        clusters = self.inner.cluster(embeddings, token_counts)
        t = self.tracker
        with t.lock:
            start, n = t.layer_start, len(embeddings)
            layer = t.layer + 1
            first_parent = start + n
            parents = []
            for j, members in enumerate(clusters):
                children = [start + i for i in members]
                pid = first_parent + j
                parents.append({"id": pid, "children": children})
                key = _context_key([t.texts.get(c, "") for c in children])
                t.pending[key] = (layer, pid)
            t.layer, t.layer_start, t.layer_count = layer, first_parent, len(clusters)
        t.job.emit("clusters", {"layer": layer, "input": n, "parents": parents})
        return clusters


class _TrackingSummarizer(BaseSummarizationModel):
    def __init__(self, inner: BaseSummarizationModel, tracker: _Tracker) -> None:
        self.inner = inner
        self.tracker = tracker
        self.model_id = inner.model_id

    def summarize(self, context: str, max_tokens: int = 150) -> str:
        out = self.inner.summarize(context, max_tokens)
        t = self.tracker
        with t.lock:
            layer, pid = t.pending.pop(context, (t.layer, -1))
            if pid >= 0:
                t.texts[pid] = out
            t.llm_calls += 1
            t.summaries_done += 1
            cost = t.cost()
        t.job.emit(
            "summary", {"layer": layer, "id": pid, "preview": preview(out, 160), "cost": cost}
        )
        return out


class _CountingEmbedder(BaseEmbeddingModel):
    def __init__(self, inner: BaseEmbeddingModel, tracker: _Tracker) -> None:
        self.inner = inner
        self.tracker = tracker
        self.model_id = inner.model_id

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        out = self.inner.embed_documents(texts)
        with self.tracker.lock:
            self.tracker.embed_calls += 1
            self.tracker.embedded_texts += len(texts)
        return out

    def embed_query(self, text: str) -> np.ndarray:
        return self.inner.embed_query(text)


def run_build(
    job: Job,
    *,
    text: str,
    name: str,
    source: dict[str, Any],
    preset: str,
    config: RaptorConfig,
    models: ModelBundle,
    index_id: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Build an index from ``text``, streaming detailed progress to ``job``, and save it."""
    tracker = _Tracker(job)
    builder = TreeBuilder(
        config,
        _CountingEmbedder(models.embedding, tracker),
        _TrackingSummarizer(models.summarization, tracker),
        chunker=_TrackingChunker(make_chunker(config.chunking), tracker),
        clusterer=_TrackingClusterer(make_clusterer(config.clustering), tracker),
    )
    job.emit("start", {"stage": "start", "name": name, "chars": len(text), "preset": preset})
    t0 = time.time()
    tree = builder.build_from_text(text, progress=lambda ev: job.emit("progress", dict(ev)))
    seconds = round(time.time() - t0, 2)
    tree.metadata.update(
        name=name,
        source=source,
        preset=preset,
        created_at=datetime.now(timezone.utc).isoformat(),
        build_seconds=seconds,
        cost=tracker.cost(),
    )
    save_tree(tree, out_dir)
    return {
        "index_id": index_id,
        "num_nodes": len(tree.all_nodes),
        "num_layers": tree.num_layers,
        "layer_counts": {str(k): len(v) for k, v in sorted(tree.layer_to_nodes.items())},
        "build_seconds": seconds,
        "cost": tracker.cost(),
    }
