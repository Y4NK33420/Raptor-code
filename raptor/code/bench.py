"""Evaluation helpers for code retrieval: SWE-bench localization and repo QA judging.

Localization measures whether the files and functions edited by an issue's gold patch
are retrieved for the issue text (retrieval only, no reader calls). Baselines:

* ``lines``      flat line-window RAG (40-line windows, 10 lines overlap)
* ``symbols``    flat symbol-chunk RAG (leaf nodes only)
* ``structural`` symbol leaves + class / file / dir / repo summaries
* ``full``       structural + semantic concept nodes

each with or without BM25 fusion and call-graph expansion.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..models.base import BaseEmbeddingModel, BaseQAModel
from .index import CodeIndex
from .languages import PythonExtractor
from .lexical import BM25, tokenize
from .prompts import JUDGE_PROMPT

KS = (1, 5, 10)
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


# ------------------------------------------------------------------ gold patches
def patch_old_lines(patch: str) -> dict[str, set[int]]:
    """Lines of each pre-patch file that a unified diff touches.

    Removed lines count; for pure insertions the old line just before the insertion point
    counts, so additions inside a function still point at that function.
    """
    out: dict[str, set[int]] = {}
    path: str | None = None
    old = 0
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            path = None
        elif line.startswith("--- "):
            src = line[4:].strip()
            path = src[2:] if src.startswith("a/") else None
            if path is not None:
                out.setdefault(path, set())
        elif line.startswith("+++ "):
            continue
        elif (m := _HUNK.match(line)) is not None:
            old = int(m.group(1))
        elif path is None:
            continue
        elif line.startswith("-"):
            out[path].add(old)
            old += 1
        elif line.startswith("+"):
            out[path].add(max(1, old - 1))
            out[path].add(old)
        elif line.startswith(" ") or line == "":
            old += 1
    return out


def gold_functions(source: bytes, path: str, lines: Iterable[int]) -> set[str]:
    """Qualified names of the innermost function/method (else class) covering ``lines``."""
    pf = PythonExtractor().parse(source, path)
    found: set[str] = set()
    for ln in lines:
        best = None
        for s in pf.symbols:
            if s.start_line <= ln <= s.end_line and (
                best is None or (s.end_line - s.start_line) < (best.end_line - best.start_line)
            ):
                best = s
        if best is not None:
            found.add(best.qualname)
    return found


# ----------------------------------------------------------------------- metrics
def recall_at(ranked: Sequence[str], gold: set[str], k: int) -> float:
    return len(gold & set(ranked[:k])) / len(gold) if gold else 0.0


def mrr(ranked: Sequence[str], gold: set[str]) -> float:
    for i, item in enumerate(ranked, 1):
        if item in gold:
            return 1.0 / i
    return 0.0


def score_ranking(ranked: Sequence[str], gold: set[str]) -> dict[str, float]:
    out = {f"R@{k}": recall_at(ranked, gold, k) for k in KS}
    out["MRR"] = mrr(ranked, gold)
    return out


def mean_scores(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def _dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


# --------------------------------------------------------------- tree baselines
VARIANT_KINDS: dict[str, tuple[str, ...]] = {
    "symbols": ("symbol",),
    "structural": ("symbol", "class", "file", "dir", "repo"),
    "full": ("symbol", "class", "file", "dir", "repo", "concept"),
}


def rank_from_index(
    index: CodeIndex,
    query: str,
    variant: str,
    *,
    bm25: bool,
    graph: bool,
    k: int = 60,
) -> tuple[list[str], list[str]]:
    """``(files, functions)`` ranked for ``query`` by one tree variant."""
    kinds = VARIANT_KINDS[variant]
    hits = index.search(
        query,
        k,
        kinds=kinds,
        bm25=bm25,
        graph_expansion=graph,
        descend=variant != "symbols",
        routing=variant != "symbols",
    )
    files: list[str] = []
    funcs: list[str] = []
    for h in hits:
        node = index.tree.all_nodes[h.node_index]
        if h.kind in ("symbol", "class", "file"):
            files.append(h.path)
        if h.kind == "symbol":
            for q in node.metadata.get("symbols") or [h.symbol]:
                if q != "<module>":
                    funcs.append(f"{h.path}::{q}")
        elif h.kind == "class":
            funcs.append(f"{h.path}::{h.symbol}")
    return _dedupe(files), _dedupe(funcs)


# ------------------------------------------------------------ line-window baseline
@dataclass
class LineWindowIndex:
    """Flat line-window RAG over the same files as a code index."""

    windows: list[tuple[str, int, int]]  # (path, start, end) 1-based inclusive
    texts: list[str]
    embeddings: np.ndarray
    symbols: dict[str, list[tuple[int, int, str]]] = field(default_factory=dict)
    _bm25: BM25 | None = None

    @classmethod
    def build(
        cls,
        index: CodeIndex,
        embedding_model: BaseEmbeddingModel,
        *,
        window: int = 40,
        overlap: int = 10,
    ) -> LineWindowIndex:
        windows, texts = [], []
        symbols: dict[str, list[tuple[int, int, str]]] = {}
        step = max(1, window - overlap)
        for path in index.files():
            src = (index.repo_path / path).read_text(encoding="utf-8", errors="replace")
            lines = src.splitlines()
            parsed = index._files[path]["parsed"]
            symbols[path] = [
                (s["start_line"], s["end_line"], s["qualname"]) for s in parsed["symbols"]
            ]
            for start in range(0, max(1, len(lines)), step):
                chunk = lines[start : start + window]
                if not any(x.strip() for x in chunk):
                    continue
                windows.append((path, start + 1, start + len(chunk)))
                texts.append(f"# {path}:{start + 1}-{start + len(chunk)}\n" + "\n".join(chunk))
                if start + window >= len(lines):
                    break
        emb = np.asarray(embedding_model.embed_documents(texts), dtype=np.float32)
        emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
        return cls(windows, texts, emb, symbols)

    def rank(
        self, query: str, embedding_model: BaseEmbeddingModel, *, bm25: bool, k: int = 60
    ) -> tuple[list[str], list[str]]:
        q = np.asarray(embedding_model.embed_query(query[:8000]), dtype=np.float32)
        q /= np.linalg.norm(q) or 1.0
        sims = self.embeddings @ q
        dense_rank = [int(i) for i in np.argsort(-sims)[: k * 2]]
        fused: dict[int, float] = {}
        for r, i in enumerate(dense_rank):
            fused[i] = fused.get(i, 0.0) + 1 / (60 + r + 1)
        if bm25:
            if self._bm25 is None:
                self._bm25 = BM25([tokenize(t) for t in self.texts])
            for r, (i, _) in enumerate(self._bm25.top(tokenize(query), k * 2)):
                fused[i] = fused.get(i, 0.0) + 1 / (60 + r + 1)
        order = sorted(fused, key=lambda i: (-fused[i], i))[:k]
        files, funcs = [], []
        for i in order:
            path, a, b = self.windows[i]
            files.append(path)
            # Symbols overlapping the window, most specific (shortest span) first.
            spans = sorted(
                (e - s, qn) for s, e, qn in self.symbols.get(path, []) if s <= b and e >= a
            )
            funcs.extend(f"{path}::{qn}" for _, qn in spans)
        return _dedupe(files), _dedupe(funcs)


# ------------------------------------------------------------------- QA judging
def judge_answer(
    judge: BaseQAModel, question: str, reference: str, answer: str
) -> dict[str, Any]:
    """Score ``answer`` against ``reference`` with an LLM-judge rubric (1-5 per criterion)."""
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)
    raw = judge.answer_question(prompt, "Return the JSON grade.")
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        data = {}
    out: dict[str, Any] = {"raw": raw}
    for key in ("correctness", "completeness", "grounding"):
        try:
            out[key] = float(data.get(key))
        except (TypeError, ValueError):
            out[key] = float("nan")
    out["comment"] = data.get("comment", "")
    return out
