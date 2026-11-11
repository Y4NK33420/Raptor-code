"""Chunking strategies. Register new ones with ``@CHUNKERS.register("name")``."""

from ..config import ChunkingConfig
from .base import CHUNKERS, Chunk, Chunker, default_tokenizer
from .sentence import SentenceChunker, sentence_spans, split_text

__all__ = [
    "CHUNKERS",
    "Chunk",
    "Chunker",
    "SentenceChunker",
    "default_tokenizer",
    "make_chunker",
    "sentence_spans",
    "split_text",
]


def make_chunker(cfg: ChunkingConfig) -> Chunker:
    return CHUNKERS.create(
        cfg.strategy, max_tokens=cfg.max_tokens, overlap_tokens=cfg.overlap_tokens, **cfg.params
    )
