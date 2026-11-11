"""Persistent on-disk cache for model calls.

Experiments re-run the same embeddings and summaries many times; caching them makes
re-runs free and deterministic. Keys hash the model id, call parameters and input
text, so changing any of them is a cache miss.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseEmbeddingModel, BaseQAModel, BaseSummarizationModel


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}


class DiskCache:
    """A small SQLite key/value store (safe to share across threads)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v BLOB NOT NULL)")
        self._conn.commit()

    @staticmethod
    def key(*parts: Any) -> str:
        blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str).encode()
        return hashlib.sha256(blob).hexdigest()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            row = self._conn.execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        return None if row is None else bytes(row[0])

    def get_many(self, keys: Sequence[str]) -> dict[str, bytes]:
        out: dict[str, bytes] = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                chunk = list(keys[i : i + 500])
                q = "SELECT k, v FROM kv WHERE k IN ({})".format(",".join("?" * len(chunk)))
                for k, v in self._conn.execute(q, chunk):
                    out[k] = bytes(v)
        return out

    def set_many(self, items: dict[str, bytes]) -> None:
        if not items:
            return
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", items.items())
            self._conn.commit()

    def set(self, key: str, value: bytes) -> None:
        self.set_many({key: value})

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_CACHES: dict[Path, DiskCache] = {}
_CACHES_LOCK = threading.Lock()


def open_cache(directory: str | Path) -> DiskCache:
    """Process-wide cache instance for ``directory``."""
    path = Path(directory).resolve() / "cache.sqlite"
    with _CACHES_LOCK:
        if path not in _CACHES:
            _CACHES[path] = DiskCache(path)
        return _CACHES[path]


# ----------------------------------------------------------------------- wrappers
class CachedEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, inner: BaseEmbeddingModel, cache: DiskCache) -> None:
        self.inner = inner
        self.cache = cache
        self.stats = CacheStats()
        self.model_id = inner.model_id

    def _lookup(self, kind: str, texts: Sequence[str], compute) -> np.ndarray:
        keys = [DiskCache.key("emb", self.inner.model_id, kind, t) for t in texts]
        found = self.cache.get_many(list(dict.fromkeys(keys)))
        missing: dict[str, int] = {}  # unique missing key -> first position
        for i, k in enumerate(keys):
            if k not in found and k not in missing:
                missing[k] = i
        self.stats.misses += len(missing)
        self.stats.hits += len(texts) - len(missing)
        if missing:
            computed = np.asarray(compute([texts[i] for i in missing.values()]), dtype=np.float32)
            new = {k: computed[j].tobytes() for j, k in enumerate(missing)}
            self.cache.set_many(new)
            found.update(new)
        return np.stack([np.frombuffer(found[k], dtype=np.float32) for k in keys])

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return self._lookup("doc", texts, self.inner.embed_documents)

    def embed_query(self, text: str) -> np.ndarray:
        return self._lookup("query", [text], lambda ts: [self.inner.embed_query(ts[0])])[0]


class CachedSummarizationModel(BaseSummarizationModel):
    def __init__(self, inner: BaseSummarizationModel, cache: DiskCache) -> None:
        self.inner = inner
        self.cache = cache
        self.stats = CacheStats()
        self.model_id = inner.model_id

    def summarize(self, context: str, max_tokens: int = 150) -> str:
        key = DiskCache.key("sum", self.inner.model_id, max_tokens, context)
        hit = self.cache.get(key)
        if hit is not None:
            self.stats.hits += 1
            return hit.decode("utf-8")
        self.stats.misses += 1
        out = self.inner.summarize(context, max_tokens)
        self.cache.set(key, out.encode("utf-8"))
        return out


class CachedQAModel(BaseQAModel):
    def __init__(self, inner: BaseQAModel, cache: DiskCache) -> None:
        self.inner = inner
        self.cache = cache
        self.stats = CacheStats()
        self.model_id = inner.model_id

    def answer_question(self, context: str, question: str) -> str:
        key = DiskCache.key("qa", self.inner.model_id, context, question)
        hit = self.cache.get(key)
        if hit is not None:
            self.stats.hits += 1
            return hit.decode("utf-8")
        self.stats.misses += 1
        out = self.inner.answer_question(context, question)
        self.cache.set(key, out.encode("utf-8"))
        return out

    def stream_answer(self, context: str, question: str) -> Iterator[str]:
        key = DiskCache.key("qa", self.inner.model_id, context, question)
        hit = self.cache.get(key)
        if hit is not None:
            self.stats.hits += 1
            yield hit.decode("utf-8")
            return
        self.stats.misses += 1
        parts: list[str] = []
        for piece in self.inner.stream_answer(context, question):
            parts.append(piece)
            yield piece
        self.cache.set(key, "".join(parts).encode("utf-8"))

    def __getattr__(self, name: str) -> Any:
        # Expose extra methods of the wrapped model (e.g. structured MCQ answering).
        return getattr(self.inner, name)
