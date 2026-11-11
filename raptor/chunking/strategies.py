"""Alternative chunking strategies: fixed token windows, paragraphs, recursive, semantic.

All of them return chunks that are exact slices of the source text (with offsets) and
never exceed ``max_tokens``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import tiktoken

from .base import CHUNKERS, Chunk, Chunker, default_tokenizer
from .sentence import SentenceChunker, _trim

if TYPE_CHECKING:
    from ..models.base import BaseEmbeddingModel

Unit = tuple[int, int, int]  # (start, end, n_tokens)

# Paragraph break: a blank line, or a newline followed directly by text. The second form
# matters for hard-wrapped sources (like QuALITY) where wrapped lines start with a space.
PARAGRAPH_BREAK = r"\n[ \t]*\n\s*|\n(?=\S)"
# Sentence end inside a paragraph (single newlines are treated as spaces).
_SENTENCE_END = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s)")


def _spans_between(
    text: str, pattern: re.Pattern[str], start: int, end: int
) -> list[tuple[int, int]]:
    spans, pos = [], start
    for m in pattern.finditer(text, start, end):
        if m.end() == m.start():
            continue
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, end))
    return [s for s in (_trim(text, a, b) for a, b in spans) if s is not None]


def paragraph_spans(text: str, pattern: str = PARAGRAPH_BREAK) -> list[tuple[int, int]]:
    return _spans_between(text, re.compile(pattern), 0, len(text))


def prose_sentence_spans(text: str, pattern: str = PARAGRAPH_BREAK) -> list[tuple[int, int]]:
    """Sentences that may span wrapped lines; paragraph breaks always end a sentence."""
    out: list[tuple[int, int]] = []
    for ps, pe in paragraph_spans(text, pattern):
        pos = ps
        for m in _SENTENCE_END.finditer(text, ps, pe):
            span = _trim(text, pos, m.end())
            if span:
                out.append(span)
            pos = m.end()
        span = _trim(text, pos, pe)
        if span:
            out.append(span)
    return out


class _UnitChunker(SentenceChunker):
    """Greedy packing (with overlap) over strategy-specific units; see SentenceChunker.split."""

    def _fit(self, text: str, start: int, end: int) -> list[Unit]:
        """Split ``text[start:end]`` into units that each fit ``max_tokens``."""
        n = self._count(text[start:end])
        if n <= self.max_tokens:
            return [(start, end, n)]
        units: list[Unit] = []
        for s, e in sentence_spans_in(text, start, end):
            m = self._count(text[s:e])
            if m <= self.max_tokens:
                units.append((s, e, m))
            else:
                units.extend(_shift(SentenceChunker._units(self, text[s:e]), s))
        return units


def sentence_spans_in(text: str, start: int, end: int) -> list[tuple[int, int]]:
    return [(start + a, start + b) for a, b in prose_sentence_spans(text[start:end])]


def _shift(units: list[Unit], offset: int) -> list[Unit]:
    return [(s + offset, e + offset, n) for s, e, n in units]


class ParagraphChunker(_UnitChunker):
    """Packs whole paragraphs; a paragraph is split (by sentences) only if it is too long."""

    def __init__(self, *args: object, pattern: str = PARAGRAPH_BREAK, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.pattern = pattern

    def _units(self, text: str) -> list[Unit]:
        units: list[Unit] = []
        for s, e in paragraph_spans(text, self.pattern):
            units.extend(self._fit(text, s, e))
        return units


class RecursiveChunker(SentenceChunker):
    """LangChain-style recursive splitting: try coarse separators first, then finer ones.

    Pieces that fit ``max_tokens`` become units, which are then packed greedily.
    """

    DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ")

    def __init__(
        self, *args: object, separators: tuple[str, ...] | list[str] | None = None, **kwargs: object
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.separators = tuple(separators or self.DEFAULT_SEPARATORS)

    def _units(self, text: str) -> list[Unit]:
        span = _trim(text, 0, len(text))
        return self._split(text, span[0], span[1], 0) if span else []

    def _split(self, text: str, start: int, end: int, level: int) -> list[Unit]:
        n = self._count(text[start:end])
        if n <= self.max_tokens:
            return [(start, end, n)]
        if level >= len(self.separators):
            return self._token_windows(text, start, end)
        sep = self.separators[level]
        pieces: list[tuple[int, int]] = []
        pos = start
        while True:
            i = text.find(sep, pos, end)
            if i < 0:
                break
            cut = i + len(sep.rstrip()) if sep.strip() else i  # keep the '.' of '. '
            pieces.append((pos, cut))
            pos = i + len(sep)
        pieces.append((pos, end))
        trimmed = [p for p in (_trim(text, a, b) for a, b in pieces) if p is not None]
        if len(trimmed) <= 1:
            return self._split(text, start, end, level + 1)
        out: list[Unit] = []
        for a, b in trimmed:
            out.extend(self._split(text, a, b, level + 1))
        return out


class FixedTokenChunker(Chunker):
    """Windows of ``max_tokens`` tokens, each overlapping the previous by ``overlap_tokens``."""

    def __init__(
        self,
        max_tokens: int = 100,
        overlap_tokens: int = 0,
        tokenizer: tiktoken.Encoding | None = None,
    ) -> None:
        if max_tokens < 1 or not 0 <= overlap_tokens < max_tokens:
            raise ValueError("need max_tokens >= 1 and 0 <= overlap_tokens < max_tokens")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tokenizer or default_tokenizer()

    def _count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def split(self, text: str) -> list[Chunk]:
        tokens = self.tokenizer.encode(text, disallowed_special=())
        if not tokens:
            return []
        decoded, offsets = self.tokenizer.decode_with_offsets(tokens)
        if decoded != text:  # pragma: no cover - only for text tiktoken cannot round-trip
            step = self.max_tokens - self.overlap_tokens
            pieces = (
                self.tokenizer.decode(tokens[i : i + self.max_tokens]).strip()
                for i in range(0, len(tokens), step)
            )
            return [Chunk(p) for p in pieces if p]
        n = len(tokens)
        bounds = [*offsets, len(text)]
        chunks: list[Chunk] = []
        i = 0
        while i < n:
            j = min(i + self.max_tokens, n)
            span = _trim(text, bounds[i], bounds[j])
            # Trimming can change how the edges tokenize; shrink the window until it fits.
            while span and j > i + 1 and self._count(text[span[0] : span[1]]) > self.max_tokens:
                j -= 1
                span = _trim(text, bounds[i], bounds[j])
            if span:
                chunks.append(Chunk(text[span[0] : span[1]], span[0], span[1]))
            if j >= n:
                break
            i = max(j - self.overlap_tokens, i + 1)
        return chunks


class SemanticChunker(SentenceChunker):
    """Split where the embedding similarity between neighbouring sentences drops.

    ``overlap_tokens`` is ignored: segments follow topic shifts, not a sliding window.

    Breakpoints are the gaps whose cosine distance is above the ``breakpoint_percentile``
    of all gaps. Segments longer than ``max_tokens`` are re-packed by sentences, and
    segments shorter than ``min_tokens`` are merged into a neighbour.
    """

    def __init__(
        self,
        max_tokens: int = 100,
        overlap_tokens: int = 0,
        tokenizer: tiktoken.Encoding | None = None,
        *,
        embedding_model: BaseEmbeddingModel | None = None,
        breakpoint_percentile: float = 80.0,
        min_tokens: int | None = None,
    ) -> None:
        super().__init__(max_tokens, overlap_tokens, tokenizer)
        if embedding_model is None:
            raise ValueError("the semantic chunker needs an embedding model")
        self.embedding_model = embedding_model
        self.breakpoint_percentile = breakpoint_percentile
        self.min_tokens = max_tokens // 4 if min_tokens is None else min_tokens

    def split(self, text: str) -> list[Chunk]:
        sentences = prose_sentence_spans(text)
        if not sentences:
            return []
        units: list[Unit] = []
        for s, e in sentences:  # sentence-level units, each <= max_tokens
            n = self._count(text[s:e])
            if n <= self.max_tokens:
                units.append((s, e, n))
            else:
                units.extend(_shift(SentenceChunker._units(self, text[s:e]), s))
        if len(units) == 1:
            return [self._make(text, units)]
        emb = np.asarray(
            self.embedding_model.embed_documents([text[s:e] for s, e, _ in units]), dtype=np.float32
        )
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms
        distance = 1.0 - np.sum(emb[:-1] * emb[1:], axis=1)
        cut_after = set(np.where(distance > np.percentile(distance, self.breakpoint_percentile))[0])

        segments: list[list[Unit]] = [[]]
        for i, u in enumerate(units):
            segments[-1].append(u)
            if i in cut_after:
                segments.append([])
        segments = [s for s in segments if s]
        segments = self._merge_small(segments)

        chunks: list[Chunk] = []
        for seg in segments:
            if sum(u[2] for u in seg) <= self.max_tokens:
                chunks.append(self._make(text, seg))
            else:
                chunks.extend(self._pack(text, seg))
        return chunks

    def _merge_small(self, segments: list[list[Unit]]) -> list[list[Unit]]:
        out: list[list[Unit]] = []
        for seg in segments:
            size = sum(u[2] for u in seg)
            if (
                out
                and (size < self.min_tokens or sum(u[2] for u in out[-1]) < self.min_tokens)
                and sum(u[2] for u in out[-1]) + size <= self.max_tokens
            ):
                out[-1] = out[-1] + seg
            else:
                out.append(seg)
        return out

    def _pack(self, text: str, units: list[Unit]) -> list[Chunk]:
        chunks: list[Chunk] = []
        current: list[Unit] = []
        length = 0
        for u in units:
            if current and length + u[2] > self.max_tokens:
                chunks.append(self._make(text, current))
                current, length = [], 0
            current.append(u)
            length += u[2]
        if current:
            chunks.append(self._make(text, current))
        return chunks


@CHUNKERS.register("fixed_tokens")
def _make_fixed(max_tokens: int = 100, overlap_tokens: int = 0, **_: object) -> Chunker:
    return FixedTokenChunker(max_tokens=max_tokens, overlap_tokens=overlap_tokens)


@CHUNKERS.register("paragraph")
def _make_paragraph(
    max_tokens: int = 100, overlap_tokens: int = 0, pattern: str = PARAGRAPH_BREAK, **_: object
) -> Chunker:
    return ParagraphChunker(max_tokens=max_tokens, overlap_tokens=overlap_tokens, pattern=pattern)


@CHUNKERS.register("recursive")
def _make_recursive(
    max_tokens: int = 100,
    overlap_tokens: int = 0,
    separators: list[str] | None = None,
    **_: object,
) -> Chunker:
    return RecursiveChunker(
        max_tokens=max_tokens, overlap_tokens=overlap_tokens, separators=separators
    )


@CHUNKERS.register("semantic")
def _make_semantic(
    max_tokens: int = 100,
    overlap_tokens: int = 0,
    embedding_model: BaseEmbeddingModel | None = None,
    breakpoint_percentile: float = 80.0,
    min_tokens: int | None = None,
    **_: object,
) -> Chunker:
    return SemanticChunker(
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        embedding_model=embedding_model,
        breakpoint_percentile=breakpoint_percentile,
        min_tokens=min_tokens,
    )
