"""BM25 over code identifiers and docstrings."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

_STOPWORDS = (
    "a an and are as at be but by for from has have if in into is it its of on or that the "
    "their then there these this to was were will with not no can how what where why which "
    "who when does do self cls def return none true false import class pass raise else elif "
    "try except finally while lambda yield"
)
STOPWORDS = frozenset(_STOPWORDS.split())


def split_identifier(word: str) -> list[str]:
    """``getHTTPResponse_code`` -> ``["get", "http", "response", "code"]``."""
    out: list[str] = []
    for piece in word.split("_"):
        if piece:
            out.extend(m.lower() for m in _CAMEL.findall(piece))
    return out


def tokenize(text: str) -> list[str]:
    """Identifier-aware tokens: each identifier, lowercased, plus its camel/snake parts."""
    tokens: list[str] = []
    for word in _IDENT.findall(text):
        low = word.lower()
        parts = split_identifier(word)
        if len(parts) > 1 or (parts and parts[0] != low):
            tokens.append(low)
            tokens.extend(p for p in parts if p not in STOPWORDS and len(p) > 1)
        elif low not in STOPWORDS and len(low) > 1:
            tokens.append(low)
    return tokens


class BM25:
    """Okapi BM25 with sparse postings (fast enough for repo-sized corpora)."""

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.lengths = [len(d) for d in docs]
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.postings.setdefault(term, []).append((i, tf))
        self.idf = {
            t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
            for t, p in self.postings.items()
        }

    def scores(self, query: Sequence[str]) -> dict[int, float]:
        out: dict[int, float] = {}
        if not self.n:
            return out
        for term, qtf in Counter(query).items():
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf[term]
            for i, tf in plist:
                denom = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avgdl or 1))
                out[i] = out.get(i, 0.0) + qtf * idf * tf * (self.k1 + 1) / denom
        return out

    def top(self, query: Sequence[str], k: int) -> list[tuple[int, float]]:
        s = self.scores(query)
        return sorted(s.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
