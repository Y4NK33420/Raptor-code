"""Local models (no API quota): sentence-transformers embeddings and a cross-encoder reranker."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base import BaseEmbeddingModel


class SBertEmbeddingModel(BaseEmbeddingModel):
    """Sentence-transformers embedder.

    For instruction-style models set the prefixes, e.g. e5: ``query_prefix="query: "``,
    ``document_prefix="passage: "``; bge: ``query_prefix="Represent this sentence for
    searching relevant passages: "``.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        *,
        device: str | None = None,
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ImportError("Local embeddings need `pip install raptor-code[local]`") from exc
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size
        self.normalize = normalize
        self.model = SentenceTransformer(model_name, device=device)
        self.model_id = f"sbert:{model_name}:q={query_prefix!r}:d={document_prefix!r}"

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])[0]


class CrossEncoderReranker:
    """Scores (query, passage) pairs with a local cross-encoder; higher is more relevant."""

    def __init__(
        self, model_name: str = "BAAI/bge-reranker-base", *, device: str | None = None
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ImportError("Reranking needs `pip install raptor-code[local]`") from exc
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device)

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(self.model.predict([(query, p) for p in passages]), dtype=np.float32)
