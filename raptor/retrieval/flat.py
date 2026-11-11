"""Flat RAG baseline: dense retrieval over leaf chunks only (no hierarchy)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from ..chunking import Chunker, SentenceChunker
from ..models.base import BaseEmbeddingModel, BaseQAModel
from ..types import Tree
from .base import unit

logger = logging.getLogger(__name__)


class FlatRAG:
    """Embed chunks, retrieve the top-k by cosine similarity, answer from the raw chunks.

    ``chunks`` and ``embeddings`` are public so a FlatRAG can share the exact leaf chunks
    of a RAPTOR tree (see :meth:`from_tree`), which makes comparisons fair.
    """

    separator = "\n\n---\n\n"

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        qa_model: BaseQAModel | None = None,
        max_tokens: int = 250,
        top_k: int = 5,
        chunker: Chunker | None = None,
        tokenizer=None,
    ) -> None:
        self.embedding_model = embedding_model
        self.qa_model = qa_model
        self.top_k = top_k
        self.chunker = chunker or SentenceChunker(max_tokens, tokenizer=tokenizer)
        self.chunks: list[str] = []
        self.embeddings: Sequence[Sequence[float]] | np.ndarray = []

    @classmethod
    def from_tree(
        cls,
        tree: Tree,
        embedding_model: BaseEmbeddingModel,
        qa_model: BaseQAModel | None = None,
        top_k: int = 5,
    ) -> FlatRAG:
        flat = cls(embedding_model, qa_model, top_k=top_k)
        leaves = [tree.leaf_nodes[i] for i in sorted(tree.leaf_nodes)]
        flat.chunks = [n.text for n in leaves]
        flat.embeddings = np.stack([np.asarray(n.embeddings[tree.embedding_key]) for n in leaves])
        return flat

    def add_documents(self, text: str) -> None:
        self.chunks = self.chunker.split_texts(text)
        self.embeddings = self.embedding_model.embed_documents(self.chunks)
        logger.info("Flat RAG indexed %d chunks", len(self.chunks))

    def retrieve(self, question: str, top_k: int | None = None) -> tuple[str, list[int]]:
        k = top_k or self.top_k
        if not self.chunks:
            raise ValueError("FlatRAG has no chunks; call add_documents() or from_tree() first")
        matrix = np.asarray(self.embeddings, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sims = (matrix / norms) @ unit(self.embedding_model.embed_query(question))
        top = [int(i) for i in np.argsort(-sims, kind="stable")[:k]]
        return self.separator.join(self.chunks[i] for i in top), top

    def answer_question(self, question: str, top_k: int | None = None) -> str:
        if self.qa_model is None:
            raise ValueError("FlatRAG was created without a qa_model")
        context, _ = self.retrieve(question, top_k)
        return self.qa_model.answer_question(context, question)
