"""Sentence-greedy chunking (the original RAPTOR strategy, with bugs fixed).

Sentences are packed into chunks of at most ``max_tokens`` tokens. Fixes over the
original ``split_text``:

* sentence-ending punctuation is kept (the old regex split *on* ``.!?`` and dropped them);
* chunk text is the exact source slice, so offsets are reliable;
* ``overlap_tokens`` really is a token budget (old code mixed chunk-local and global
  indices when computing the carried-over length);
* sentences longer than ``max_tokens`` are split on clause punctuation and then, if
  still too long, by token windows, so no chunk exceeds the limit.
"""

from __future__ import annotations

import re

import tiktoken

from .base import CHUNKERS, Chunk, Chunker, default_tokenizer

# A sentence ends at .!? (plus any closing quotes/brackets) followed by whitespace, or at a newline.
_BOUNDARY = re.compile(r"([.!?]+[\"'”’)\]]*)(\s+)|\n+")
_CLAUSE = re.compile(r"(?<=[,;:])\s+")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` character spans of the sentences in ``text`` (whitespace-trimmed)."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _BOUNDARY.finditer(text):
        end = m.end(1) if m.group(1) else m.start()
        spans.append((pos, end))
        pos = m.end()
    spans.append((pos, len(text)))
    return [s for s in (_trim(text, a, b) for a, b in spans) if s is not None]


def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


class SentenceChunker(Chunker):
    def __init__(
        self,
        max_tokens: int = 100,
        overlap_tokens: int = 0,
        tokenizer: tiktoken.Encoding | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if overlap_tokens < 0 or overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be in [0, max_tokens)")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tokenizer or default_tokenizer()

    def _count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def _units(self, text: str) -> list[tuple[int, int, int]]:
        """Sentence (or sub-sentence) spans with token counts, each <= max_tokens."""
        units: list[tuple[int, int, int]] = []
        for start, end in sentence_spans(text):
            n = self._count(text[start:end])
            if n <= self.max_tokens:
                units.append((start, end, n))
                continue
            for cs, ce in self._clause_spans(text, start, end):
                cn = self._count(text[cs:ce])
                if cn <= self.max_tokens:
                    units.append((cs, ce, cn))
                else:
                    units.extend(self._token_windows(text, cs, ce))
        return units

    @staticmethod
    def _clause_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
        spans, pos = [], start
        for m in _CLAUSE.finditer(text, start, end):
            spans.append((pos, m.start()))
            pos = m.end()
        spans.append((pos, end))
        return [s for s in (_trim(text, a, b) for a, b in spans) if s is not None]

    def _token_windows(self, text: str, start: int, end: int) -> list[tuple[int, int, int]]:
        tokens = self.tokenizer.encode(text[start:end], disallowed_special=())
        out = []
        offset = start
        for i in range(0, len(tokens), self.max_tokens):
            piece = self.tokenizer.decode(tokens[i : i + self.max_tokens])
            # Map decoded piece back onto the source (decoding can normalise bytes).
            found = text.find(piece, offset, end) if piece else -1
            s = found if found >= 0 else offset
            e = min(end, s + len(piece))
            if e > s:
                out.append((s, e, len(tokens[i : i + self.max_tokens])))
            offset = e
        return out

    def split(self, text: str) -> list[Chunk]:
        units = self._units(text)
        chunks: list[Chunk] = []
        current: list[tuple[int, int, int]] = []
        length = 0
        for unit in units:
            if current and length + unit[2] > self.max_tokens:
                chunks.append(self._make(text, current))
                current = self._overlap_tail(current)
                length = sum(u[2] for u in current)
                # The carried tail plus this unit might still not fit; drop tail if so.
                if length + unit[2] > self.max_tokens:
                    current, length = [], 0
            current.append(unit)
            length += unit[2]
        if current:
            chunks.append(self._make(text, current))
        return chunks

    def _overlap_tail(self, units: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        if self.overlap_tokens == 0:
            return []
        tail: list[tuple[int, int, int]] = []
        total = 0
        for u in reversed(units):
            if total + u[2] > self.overlap_tokens:
                break
            tail.insert(0, u)
            total += u[2]
        return tail

    @staticmethod
    def _make(text: str, units: list[tuple[int, int, int]]) -> Chunk:
        start, end = units[0][0], units[-1][1]
        return Chunk(text=text[start:end], start=start, end=end)


@CHUNKERS.register("sentence")
def _make_sentence(max_tokens: int = 100, overlap_tokens: int = 0, **_: object) -> Chunker:
    return SentenceChunker(max_tokens=max_tokens, overlap_tokens=overlap_tokens)


def split_text(
    text: str, tokenizer: tiktoken.Encoding, max_tokens: int, overlap: int = 0
) -> list[str]:
    """Backward-compatible wrapper for the original ``raptor.utils.split_text``."""
    return SentenceChunker(max_tokens, overlap_tokens=overlap, tokenizer=tokenizer).split_texts(
        text
    )
