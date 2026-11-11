"""Index registry, per-index model resolution, and retrieval helpers used by the API."""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..chunking import default_tokenizer
from ..config import RetrievalConfig
from ..models import ModelBundle
from ..models.base import BaseEmbeddingModel
from ..retrieval import RetrievalResult, TreeRetriever
from ..retrieval.base import unit
from ..storage import TREE_FILE, load_tree
from ..types import Node, Tree

logger = logging.getLogger(__name__)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,120}$")
_DEMO_QUALITY = re.compile(r"^quality_story_(\d+)")


def preview(text: str, n: int = 220) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= n else flat[: n - 1].rstrip() + "…"


def slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", name.lower()).strip("-")
    return (s or "index")[:48]


@dataclass
class IndexEntry:
    id: str
    path: Path
    origin: str  # "demo" | "user"

    @property
    def mtime(self) -> float:
        return (self.path / TREE_FILE).stat().st_mtime


class IndexRegistry:
    """Discovers saved indexes under the demo and user directories and caches loaded trees."""

    def __init__(
        self,
        demo_dir: Path,
        user_dir: Path,
        *,
        quality_lookup: Callable[[int], dict[str, Any] | None] | None = None,
        max_loaded: int = 8,
    ) -> None:
        self.demo_dir = demo_dir
        self.user_dir = user_dir
        self.quality_lookup = quality_lookup
        self._trees: OrderedDict[str, tuple[float, Tree]] = OrderedDict()
        self._info: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = threading.RLock()
        self.max_loaded = max_loaded

    # ----------------------------------------------------------------- discover
    def entries(self) -> list[IndexEntry]:
        out: dict[str, IndexEntry] = {}
        for origin, root in (("demo", self.demo_dir), ("user", self.user_dir)):
            if not root.is_dir():
                continue
            for d in sorted(root.iterdir()):
                if d.is_dir() and (d / TREE_FILE).is_file() and _SAFE_ID.match(d.name):
                    out.setdefault(d.name, IndexEntry(d.name, d, origin))
        return list(out.values())

    def entry(self, index_id: str) -> IndexEntry:
        if not _SAFE_ID.match(index_id):
            raise KeyError(index_id)
        for e in self.entries():
            if e.id == index_id:
                return e
        raise KeyError(index_id)

    def new_path(self, name: str, suffix: str) -> tuple[str, Path]:
        index_id = f"{slugify(name)}-{suffix}"
        return index_id, self.user_dir / index_id

    def delete(self, index_id: str) -> None:
        e = self.entry(index_id)
        if e.origin != "user":
            raise PermissionError("demo indexes are read-only")
        with self._lock:
            self._trees.pop(index_id, None)
            self._info.pop(index_id, None)
        shutil.rmtree(e.path)

    # --------------------------------------------------------------------- load
    def tree(self, index_id: str) -> Tree:
        e = self.entry(index_id)
        mtime = e.mtime
        with self._lock:
            cached = self._trees.get(index_id)
            if cached and cached[0] == mtime:
                self._trees.move_to_end(index_id)
                return cached[1]
        tree = load_tree(e.path)
        with self._lock:
            self._trees[index_id] = (mtime, tree)
            while len(self._trees) > self.max_loaded:
                self._trees.popitem(last=False)
        return tree

    def info(self, index_id: str) -> dict[str, Any]:
        e = self.entry(index_id)
        mtime = e.mtime
        with self._lock:
            cached = self._info.get(index_id)
            if cached and cached[0] == mtime:
                return cached[1]
        doc = json.loads((e.path / TREE_FILE).read_text(encoding="utf-8"))
        meta: dict[str, Any] = doc.get("metadata", {})
        layers = {int(k): len(v) for k, v in doc.get("layers", {}).items()}
        source = meta.get("source")
        if not isinstance(source, dict):
            source = {"type": "text", "label": source} if source else {"type": "text"}
            m = _DEMO_QUALITY.match(e.id)
            if m and self.quality_lookup:
                found = self.quality_lookup(int(m.group(1)) - 1)
                if found:
                    source = {"type": "quality", **found}
        leaf_chars = sum(len(n["text"]) for n in doc["nodes"] if n.get("layer", 0) == 0)
        created = meta.get("created_at") or datetime.fromtimestamp(mtime, timezone.utc).isoformat()
        info = {
            "id": e.id,
            "name": meta.get("name") or _pretty(e.id),
            "origin": e.origin,
            "kind": "document",
            "num_nodes": len(doc["nodes"]),
            "num_layers": int(doc.get("num_layers", max(layers, default=0))),
            "layer_counts": {str(k): v for k, v in sorted(layers.items())},
            "leaf_chars": leaf_chars,
            "embedding_model": meta.get("embedding_model"),
            "summarization_model": meta.get("summarization_model"),
            "created_at": created,
            "source": source,
            "preset": meta.get("preset"),
            "build_seconds": meta.get("build_seconds"),
        }
        with self._lock:
            self._info[index_id] = (mtime, info)
        return info


def _pretty(index_id: str) -> str:
    s = re.sub(r"[_-]+", " ", index_id)
    s = re.sub(r"\bgemini\b", "", s).strip()
    return s[:1].upper() + s[1:]


# --------------------------------------------------------------------------- models
def embedding_for_tree(tree: Tree, fallback: BaseEmbeddingModel) -> BaseEmbeddingModel:
    """The embedding model that matches how ``tree`` was embedded, when it can be rebuilt.

    Trees embedded with Gemini record ``gemini:<model>:doc=<task>:q=<task>:dim=<n>``; the
    demo trees were built without task types, so queries must use none either.
    """
    mid = str(tree.metadata.get("embedding_model") or "")
    if mid == fallback.model_id or not mid.startswith("gemini:"):
        return fallback
    parts = mid.split(":")
    fields = dict(p.split("=", 1) for p in parts[2:] if "=" in p)

    def opt(v: str | None) -> str | None:
        return None if v in (None, "None", "") else v

    from ..models import CachedEmbeddingModel
    from ..models.gemini import GeminiEmbeddingModel

    dim = opt(fields.get("dim"))
    model: BaseEmbeddingModel = GeminiEmbeddingModel(
        parts[1],
        document_task_type=opt(fields.get("doc")),
        query_task_type=opt(fields.get("q")),
        output_dimensionality=int(dim) if dim else None,
    )
    if isinstance(fallback, CachedEmbeddingModel):
        model = CachedEmbeddingModel(model, fallback.cache)
    return model


def reembedded(tree: Tree, model: BaseEmbeddingModel) -> Tree:
    """A copy of ``tree`` whose node embeddings come from ``model`` (for offline demo mode)."""
    key = tree.embedding_key
    order = sorted(tree.all_nodes)
    vectors = model.embed_documents([tree.all_nodes[i].text for i in order])
    new = {
        i: Node(
            n.text,
            n.index,
            set(n.children),
            {key: np.asarray(v, dtype=np.float32)},
            n.layer,
            n.metadata,
        )
        for i, v in zip(order, vectors, strict=True)
        for n in [tree.all_nodes[i]]
    }
    layers = {layer: [new[n.index] for n in nodes] for layer, nodes in tree.layer_to_nodes.items()}
    meta = dict(tree.metadata)
    meta["embedding_model"] = model.model_id
    return Tree.from_layers(layers, meta)


class Searcher:
    """Retrieval over one index: RAPTOR (collapsed/traversal) and the flat leaf-only baseline."""

    def __init__(self, tree: Tree, embedding: BaseEmbeddingModel) -> None:
        self.tree = tree
        self.embedding = embedding
        cfg = tree.metadata.get("config", {}).get("retrieval")
        try:
            self.config = RetrievalConfig.model_validate(cfg) if cfg else RetrievalConfig()
        except Exception:
            self.config = RetrievalConfig()
        self.tokenizer = default_tokenizer()
        self._tokens: dict[int, int] = {}
        self._leaf_ids = sorted(tree.leaf_nodes)
        key = tree.embedding_key
        leaf = np.asarray([tree.all_nodes[i].embeddings[key] for i in self._leaf_ids], np.float32)
        norms = np.linalg.norm(leaf, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._leaf_matrix = leaf / norms

    def tokens(self, index: int) -> int:
        n = self._tokens.get(index)
        if n is None:
            text = self.tree.all_nodes[index].text
            n = self._tokens[index] = len(self.tokenizer.encode(text, disallowed_special=()))
        return n

    def raptor(
        self,
        question: str,
        *,
        top_k: int | None = None,
        max_tokens: int | None = None,
        mode: str | None = None,
    ) -> RetrievalResult:
        retriever = TreeRetriever(self.tree, self.embedding, self.config, tokenizer=self.tokenizer)
        return retriever.retrieve(question, top_k=top_k, max_tokens=max_tokens, mode=mode)

    def flat(self, question: str, *, top_k: int, max_tokens: int) -> list[tuple[Node, float]]:
        """Dense retrieval over leaf chunks only, under the same top-k and token budget."""
        sims = self._leaf_matrix @ unit(self.embedding.embed_query(question))
        out: list[tuple[Node, float]] = []
        total = 0
        for pos in np.argsort(-sims, kind="stable")[:top_k]:
            idx = self._leaf_ids[int(pos)]
            n = self.tokens(idx)
            if total + n > max_tokens:
                break
            out.append((self.tree.all_nodes[idx], float(sims[pos])))
            total += n
        return out

    def describe(self, picked: list[tuple[Node, float]]) -> dict[str, Any]:
        counts: dict[str, int] = {}
        nodes = []
        for node, score in picked:
            counts[str(node.layer)] = counts.get(str(node.layer), 0) + 1
            nodes.append(
                {
                    "id": node.index,
                    "layer": node.layer,
                    "score": round(score, 4),
                    "tokens": self.tokens(node.index),
                    "preview": preview(node.text, 180),
                }
            )
        return {
            "nodes": nodes,
            "layer_counts": counts,
            "context_tokens": sum(n["tokens"] for n in nodes),
        }


class SearcherCache:
    """One :class:`Searcher` per (index, mtime, embedding model)."""

    def __init__(self, registry: IndexRegistry, models: Callable[[], ModelBundle], fake: bool):
        self.registry = registry
        self.models = models
        self.fake = fake
        self._cache: dict[str, tuple[float, Searcher]] = {}
        self._lock = threading.Lock()

    def get(self, index_id: str) -> Searcher:
        entry = self.registry.entry(index_id)
        mtime = entry.mtime
        with self._lock:
            hit = self._cache.get(index_id)
            if hit and hit[0] == mtime:
                return hit[1]
        tree = self.registry.tree(index_id)
        fallback = self.models().embedding
        built_with = str(tree.metadata.get("embedding_model") or "")
        model: BaseEmbeddingModel
        if self.fake:
            model = fallback
        elif built_with.startswith("fake:"):
            from .fakes import FakeEmbeddingModel

            model = FakeEmbeddingModel()
        else:
            model = embedding_for_tree(tree, fallback)
        if built_with != model.model_id:
            # The stored vectors come from a model we are not querying with (e.g. offline
            # demo mode over a Gemini tree): re-embed the node texts so scores are meaningful.
            tree = reembedded(tree, model)
        searcher = Searcher(tree, model)
        with self._lock:
            self._cache[index_id] = (mtime, searcher)
        return searcher
