"""Tree retrieval: collapsed-tree search and top-down traversal."""

from __future__ import annotations

import logging

import numpy as np
import tiktoken

from ..chunking import default_tokenizer
from ..config import RetrievalConfig
from ..models.base import BaseEmbeddingModel
from ..types import Tree
from .base import RetrievalResult, RetrievedNode, score_all, unit

logger = logging.getLogger(__name__)


class TreeRetriever:
    def __init__(
        self,
        tree: Tree,
        embedding_model: BaseEmbeddingModel,
        config: RetrievalConfig | None = None,
        tokenizer: tiktoken.Encoding | None = None,
    ) -> None:
        self.tree = tree
        self.embedding_model = embedding_model
        self.config = config or RetrievalConfig()
        self.tokenizer = tokenizer or default_tokenizer()
        self._token_counts: dict[int, int] = {}
        self._check_model()

    def _check_model(self) -> None:
        built_with = self.tree.metadata.get("embedding_model")
        if built_with and built_with != self.embedding_model.model_id:
            logger.warning(
                "Index was embedded with %s but queries use %s; similarities may be off.",
                built_with,
                self.embedding_model.model_id,
            )

    def _tokens(self, index: int) -> int:
        n = self._token_counts.get(index)
        if n is None:
            n = len(self.tokenizer.encode(self.tree.all_nodes[index].text, disallowed_special=()))
            self._token_counts[index] = n
        return n

    # ------------------------------------------------------------------ public
    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        max_tokens: int | None = None,
        mode: str | None = None,
        start_layer: int | None = None,
        num_layers: int | None = None,
    ) -> RetrievalResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        cfg = self.config
        top_k = top_k if top_k is not None else cfg.top_k
        max_tokens = max_tokens if max_tokens is not None else cfg.max_tokens
        mode = mode or cfg.mode
        if top_k < 1 or max_tokens < 1:
            raise ValueError("top_k and max_tokens must be >= 1")
        q = self.embedding_model.embed_query(query)
        if mode == "collapsed":
            return self.collapsed(q, top_k=top_k, max_tokens=max_tokens)
        if mode == "traversal":
            return self.traversal(
                q,
                top_k=top_k,
                start_layer=start_layer if start_layer is not None else cfg.start_layer,
                num_layers=num_layers if num_layers is not None else cfg.num_layers,
            )
        raise ValueError(f"unknown retrieval mode {mode!r}")

    def collapsed(
        self, query_embedding: np.ndarray, *, top_k: int, max_tokens: int
    ) -> RetrievalResult:
        """Rank every node in the tree together; keep the best within ``top_k`` and the token budget."""
        indices, sims = score_all(self.tree, query_embedding)
        order = self._select(sims)
        picked: list[RetrievedNode] = []
        total = 0
        for pos in order[:top_k]:
            idx = indices[pos]
            n_tok = self._tokens(idx)
            if total + n_tok > max_tokens:
                break
            picked.append(RetrievedNode(self.tree.all_nodes[idx], float(sims[pos])))
            total += n_tok
        return RetrievalResult(picked)

    def traversal(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int,
        start_layer: int | None = None,
        num_layers: int | None = None,
    ) -> RetrievalResult:
        """Pick the best nodes at ``start_layer``, then descend into their children."""
        tree = self.tree
        start = tree.num_layers if start_layer is None else start_layer
        if not 0 <= start <= tree.num_layers:
            raise ValueError(f"start_layer must be between 0 and {tree.num_layers}")
        layers = start + 1 if num_layers is None else num_layers
        if not 1 <= layers <= start + 1:
            raise ValueError("num_layers must be between 1 and start_layer + 1")

        q = unit(query_embedding)
        key = tree.embedding_key
        candidates = tree.nodes_in_layer(start)
        picked: list[RetrievedNode] = []
        for step in range(layers):
            if not candidates:
                break
            matrix = np.stack([unit(n.embeddings[key]) for n in candidates])
            sims = matrix @ q
            chosen = self._select(sims)[:top_k]
            picked.extend(RetrievedNode(candidates[i], float(sims[i])) for i in chosen)
            if step < layers - 1:
                child_ids = dict.fromkeys(c for i in chosen for c in sorted(candidates[i].children))
                candidates = [tree.all_nodes[c] for c in child_ids]
        return RetrievalResult(picked)

    def _select(self, sims: np.ndarray) -> np.ndarray:
        """Indices into ``sims`` ordered best-first, filtered by threshold if configured."""
        order = np.argsort(-sims, kind="stable")
        if self.config.selection_mode == "threshold":
            order = order[sims[order] >= self.config.threshold]
        return order
