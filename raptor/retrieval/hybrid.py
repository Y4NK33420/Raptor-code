"""Ranking helpers for collapsed retrieval: BM25, reciprocal rank fusion, MMR, reranking."""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
# A short stop list: BM25's IDF already discounts common words, this just trims noise.
_STOP_WORDS = (
    "a an and are as at be but by for from had has have he her his i in is it its of on or "
    "she that the their them they this to was were what when which who why with you"
)
_STOP = frozenset(_STOP_WORDS.split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    """Okapi BM25 over a fixed list of documents (tiny, dependency-free)."""

    def __init__(self, docs: Sequence[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lengths = np.asarray([sum(tf.values()) for tf in self.tfs], dtype=np.float64)
        self.avgdl = float(self.lengths.mean()) if len(docs) else 0.0
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.tfs), dtype=np.float64)
        if not self.tfs or self.avgdl == 0:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avgdl)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.asarray([d.get(term, 0) for d in self.tfs], dtype=np.float64)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> list[int]:
    """Fuse rankings (lists of item ids, best first) by reciprocal rank fusion."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            score[item] = score.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score, key=lambda i: (-score[i], i))


def mmr(
    candidates: Sequence[int],
    relevance: np.ndarray,
    vectors: np.ndarray,
    lam: float,
) -> list[int]:
    """Order ``candidates`` by maximal marginal relevance.

    ``relevance[i]`` and ``vectors[i]`` (unit-normalised) belong to ``candidates[i]``.
    """
    remaining = list(range(len(candidates)))
    chosen: list[int] = []
    while remaining:
        if chosen:
            sim_to_chosen = (vectors[remaining] @ vectors[chosen].T).max(axis=1)
        else:
            sim_to_chosen = np.zeros(len(remaining))
        mmr_score = lam * relevance[remaining] - (1 - lam) * sim_to_chosen
        best = remaining[int(np.argmax(mmr_score))]
        chosen.append(best)
        remaining.remove(best)
    return [candidates[i] for i in chosen]


class Reranker(Protocol):
    def score(self, query: str, passages: Sequence[str]) -> np.ndarray: ...


_RERANKERS: dict[str, Reranker] = {}
_RERANKERS_LOCK = threading.Lock()


def get_reranker(model_name: str) -> Reranker:
    """Process-wide cross-encoder for ``model_name`` (loaded once)."""
    with _RERANKERS_LOCK:
        if model_name not in _RERANKERS:
            from ..models.local import CrossEncoderReranker

            _RERANKERS[model_name] = CrossEncoderReranker(model_name)
        return _RERANKERS[model_name]
