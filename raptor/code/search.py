"""Hybrid code retrieval: BM25 + dense search fused with RRF, level routing, tree
descent and call-graph expansion."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ..models.base import BaseEmbeddingModel
from ..types import Node, Tree
from .config import CodeRetrievalConfig
from .lexical import BM25, tokenize
from .types import CodeEdge, CodeHit

MAX_QUERY_CHARS = 8000

_BROAD = re.compile(
    r"\b(how|why|where|what does|what is|overview|architecture|explain|describe|flow|design|"
    r"responsib\w*|purpose|work|works|handled?|structure|organized|relationship)\b",
    re.I,
)
_IDENT_LIKE = re.compile(
    r"[A-Za-z_]\w*_\w+|[a-z]+[A-Z]\w*|[A-Z][a-z]+[A-Z]\w*|\w+\(\)|[A-Za-z_]\w*\.[A-Za-z_]\w+|`[^`]+`"
)

#: Score multipliers per node kind for each query type.
ROUTING: dict[str, dict[str, float]] = {
    "identifier": {
        "symbol": 1.4,
        "class": 1.2,
        "file": 0.9,
        "dir": 0.6,
        "repo": 0.5,
        "concept": 0.6,
    },
    "broad": {
        "symbol": 0.85,
        "class": 1.0,
        "file": 1.15,
        "dir": 1.15,
        "repo": 1.05,
        "concept": 1.2,
    },
    "mixed": {"symbol": 1.1, "class": 1.1, "file": 1.05, "dir": 0.9, "repo": 0.8, "concept": 0.95},
    "neutral": {},
}


def node_kind(node: Node) -> str:
    return str(node.metadata.get("kind", "symbol"))


def to_hit(node: Node, score: float) -> CodeHit:
    m = node.metadata
    return CodeHit(
        node_index=node.index,
        layer=node.layer,
        kind=node_kind(node),
        path=str(m.get("path", "")),
        start_line=int(m.get("start_line", 0) or 0),
        end_line=int(m.get("end_line", 0) or 0),
        symbol=str(m.get("symbol", "")),
        score=float(score),
        text=node.text,
    )


def bm25_document(node: Node) -> list[str]:
    """Tokens indexed for ``node``: identifiers (boosted), path, docstring and text."""
    m = node.metadata
    kind = node_kind(node)
    toks: list[str] = []
    names = " ".join([str(m.get("symbol", "")), *m.get("symbols", [])])
    toks += tokenize(names) * 3
    toks += tokenize(str(m.get("path", "")).replace("/", " ").replace(".py", "")) * 2
    toks += tokenize(str(m.get("signature", "")))
    toks += tokenize(str(m.get("docstring", ""))) * 2
    if kind == "symbol":
        toks += tokenize(node.text)
        if m.get("summary"):
            toks += tokenize(m["summary"])
    else:
        toks += tokenize(str(m.get("summary") or node.text))
    return toks


@dataclass
class _Ranked:
    index: int
    score: float


class CodeSearcher:
    def __init__(
        self,
        tree: Tree,
        edges: list[CodeEdge],
        embedding_model: BaseEmbeddingModel | None,
        config: CodeRetrievalConfig,
    ) -> None:
        self.tree = tree
        self.cfg = config
        self.embedding_model = embedding_model
        self.ids = sorted(tree.all_nodes)
        self.pos = {idx: i for i, idx in enumerate(self.ids)}
        self._bm25: BM25 | None = None
        self.callees: dict[int, list[int]] = defaultdict(list)
        self.callers: dict[int, list[int]] = defaultdict(list)
        self.bases: dict[int, list[int]] = defaultdict(list)
        for e in edges:
            if e.kind == "calls":
                self.callees[e.src].append(e.dst)
                self.callers[e.dst].append(e.src)
            elif e.kind == "inherits":
                self.bases[e.src].append(e.dst)
        self.parents: dict[int, list[int]] = defaultdict(list)
        for n in tree.all_nodes.values():
            for c in n.children:
                self.parents[c].append(n.index)
        self.symbol_names = {
            str(n.metadata.get("symbol", "")).split(".")[-1].lower()
            for n in tree.all_nodes.values()
            if node_kind(n) in ("symbol", "class")
        }

    # ---------------------------------------------------------------- rankers
    @property
    def bm25(self) -> BM25:
        if self._bm25 is None:
            self._bm25 = BM25([bm25_document(self.tree.all_nodes[i]) for i in self.ids])
        return self._bm25

    def dense_scores(self, query: str) -> np.ndarray:
        assert self.embedding_model is not None
        q = np.asarray(self.embedding_model.embed_query(query[:MAX_QUERY_CHARS]), np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        _, matrix = self.tree.normalized_matrix()
        return matrix @ q

    # --------------------------------------------------------------- routing
    def query_type(self, query: str) -> str:
        words = re.findall(r"[A-Za-z_]\w*", query)
        ident = bool(_IDENT_LIKE.search(query)) or (
            len(words) <= 4 and any(w.lower() in self.symbol_names for w in words)
        )
        broad = bool(_BROAD.search(query))
        if ident and broad:
            return "mixed"
        if ident:
            return "identifier"
        if broad:
            return "broad"
        return "neutral"

    # ---------------------------------------------------------------- search
    def search(
        self,
        query: str,
        k: int = 10,
        *,
        dense: bool | None = None,
        bm25: bool | None = None,
        graph_expansion: bool | None = None,
        descend: bool | None = None,
        routing: bool | None = None,
        kinds: tuple[str, ...] | None = None,
    ) -> list[CodeHit]:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        cfg = self.cfg
        use_dense = cfg.dense if dense is None else dense
        use_bm25 = cfg.bm25 if bm25 is None else bm25
        use_graph = cfg.graph_expansion if graph_expansion is None else graph_expansion
        use_descend = cfg.descend if descend is None else descend
        use_routing = cfg.routing if routing is None else routing
        if self.embedding_model is None:
            use_dense = False
        if not (use_dense or use_bm25):
            raise ValueError("enable at least one of dense / bm25")
        allowed = {
            i
            for i in self.ids
            if node_kind(self.tree.all_nodes[i]) not in cfg.exclude_kinds
            and (kinds is None or node_kind(self.tree.all_nodes[i]) in kinds)
        }
        if not allowed:
            return []

        sims: np.ndarray | None = self.dense_scores(query) if use_dense else None
        fused: dict[int, float] = defaultdict(float)
        n_cand = max(cfg.candidates, k)
        if sims is not None:
            order = np.argsort(-sims)
            ranked = [self.ids[p] for p in order if self.ids[p] in allowed][:n_cand]
            for r, idx in enumerate(ranked):
                fused[idx] += 1.0 / (cfg.rrf_k + r + 1)
        if use_bm25:
            ranked_b = [
                i
                for i, _ in (
                    (self.ids[p], s) for p, s in self.bm25.top(tokenize(query), n_cand * 3)
                )
                if i in allowed
            ][:n_cand]
            for r, idx in enumerate(ranked_b):
                fused[idx] += 1.0 / (cfg.rrf_k + r + 1)

        weights = ROUTING[self.query_type(query)] if use_routing else {}
        scores = {
            i: s * weights.get(node_kind(self.tree.all_nodes[i]), 1.0) for i, s in fused.items()
        }

        def sim_of(idx: int) -> float:
            return float(sims[self.pos[idx]]) if sims is not None else 0.0

        top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[: max(k, 10)]
        decay = cfg.expansion_decay
        if use_descend:
            for idx, s in top:
                if node_kind(self.tree.all_nodes[idx]) in ("class", "file"):
                    for leaf in self._best_leaves(idx, sim_of, allowed, 2):
                        scores[leaf] = max(scores.get(leaf, 0.0), s * decay)
        if use_graph:
            for idx, s in top[:k]:
                for nb in self._neighbors(idx, sim_of, allowed):
                    scores[nb] = max(scores.get(nb, 0.0), s * decay * 0.95)

        final = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [to_hit(self.tree.all_nodes[i], s) for i, s in final]

    # ------------------------------------------------------------- expansion
    def _best_leaves(self, idx: int, sim_of, allowed: set[int], n: int) -> list[int]:
        leaves: list[int] = []
        stack = list(self.tree.all_nodes[idx].children)
        while stack:
            c = stack.pop()
            node = self.tree.all_nodes[c]
            if node.children:
                stack.extend(node.children)
            elif c in allowed:
                leaves.append(c)
        leaves.sort(key=lambda i: (-sim_of(i), i))
        return leaves[:n]

    def neighbors(self, idx: int) -> dict[str, list[int]]:
        """Callers, callees, base classes and parent class of node ``idx``."""
        node = self.tree.all_nodes[idx]
        parent_class = [
            p for p in self.parents.get(idx, []) if node_kind(self.tree.all_nodes[p]) == "class"
        ]
        return {
            "callees": list(dict.fromkeys(self.callees.get(idx, []))),
            "callers": list(dict.fromkeys(self.callers.get(idx, []))),
            "bases": list(dict.fromkeys(self.bases.get(idx, []))),
            "parent_class": parent_class if node_kind(node) == "symbol" else [],
        }

    def _neighbors(self, idx: int, sim_of, allowed: set[int]) -> list[int]:
        nb = self.neighbors(idx)
        cands = [
            i
            for i in nb["parent_class"] + nb["callees"] + nb["callers"] + nb["bases"]
            if i in allowed and i != idx
        ]
        cands = list(dict.fromkeys(cands))
        cands.sort(key=lambda i: (-sim_of(i), i))
        return cands[: self.cfg.expansion_per_hit]
