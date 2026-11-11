"""Chunking strategies. Register new ones with ``@CHUNKERS.register("name")``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import ChunkingConfig
from .base import CHUNKERS, Chunk, Chunker, default_tokenizer
from .sentence import SentenceChunker, sentence_spans, split_text
from .strategies import FixedTokenChunker, ParagraphChunker, RecursiveChunker, SemanticChunker

if TYPE_CHECKING:
    from ..models.base import BaseEmbeddingModel

__all__ = [
    "CHUNKERS",
    "Chunk",
    "Chunker",
    "FixedTokenChunker",
    "ParagraphChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "SentenceChunker",
    "default_tokenizer",
    "make_chunker",
    "sentence_spans",
    "split_text",
]


def make_chunker(cfg: ChunkingConfig, embedding_model: BaseEmbeddingModel | None = None) -> Chunker:
    """Build the configured chunker. ``embedding_model`` is used by embedding-aware strategies."""
    extra = {"embedding_model": embedding_model} if embedding_model is not None else {}
    return CHUNKERS.create(
        cfg.strategy,
        max_tokens=cfg.max_tokens,
        overlap_tokens=cfg.overlap_tokens,
        **extra,
        **cfg.params,
    )
