"""Chunking interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import tiktoken

from ..registry import Registry


@dataclass
class Chunk:
    """A leaf chunk. ``start``/``end`` are character offsets into the source text."""

    text: str
    start: int | None = None
    end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Chunker(ABC):
    @abstractmethod
    def split(self, text: str) -> list[Chunk]: ...

    def split_texts(self, text: str) -> list[str]:
        return [c.text for c in self.split(text)]


CHUNKERS: Registry[Chunker] = Registry("chunker")


_DEFAULT_ENCODING = None


def default_tokenizer() -> tiktoken.Encoding:
    global _DEFAULT_ENCODING
    if _DEFAULT_ENCODING is None:
        _DEFAULT_ENCODING = tiktoken.get_encoding("cl100k_base")
    return _DEFAULT_ENCODING
