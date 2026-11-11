"""Model interfaces.

The class names match the original RAPTOR so existing subclasses keep working:
a subclass that only implements ``create_embedding`` still works, and new models can
implement the batched ``embed_documents`` instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np


class BaseEmbeddingModel(ABC):  # noqa: B024 - "implement one of two" is enforced in __init_subclass__
    """Turns text into vectors. Override ``embed_documents`` or ``create_embedding``."""

    #: Stable identifier used in cache keys and index metadata.
    model_id: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        own = cls.__dict__
        inherits_impl = any(
            "embed_documents" in base.__dict__ or "create_embedding" in base.__dict__
            for base in cls.__mro__[1:]
            if base is not BaseEmbeddingModel and issubclass(base, BaseEmbeddingModel)
        )
        if "embed_documents" not in own and "create_embedding" not in own and not inherits_impl:
            raise TypeError(f"{cls.__name__} must implement embed_documents or create_embedding")
        if not cls.__dict__.get("model_id"):
            cls.model_id = cls.model_id or cls.__name__

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts to be stored in the index. Returns a ``(len(texts), dim)`` array."""
        if type(self).create_embedding is BaseEmbeddingModel.create_embedding:
            raise NotImplementedError
        return np.asarray([self.create_embedding(t) for t in texts], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query. Defaults to document embedding."""
        return np.asarray(self.embed_documents([text])[0], dtype=np.float32)

    def create_embedding(self, text: str) -> list[float]:
        """Original RAPTOR single-text API."""
        return [float(x) for x in self.embed_documents([text])[0]]


class BaseSummarizationModel(ABC):
    model_id: str = ""

    @abstractmethod
    def summarize(self, context: str, max_tokens: int = 150) -> str: ...


class BaseQAModel(ABC):
    model_id: str = ""

    @abstractmethod
    def answer_question(self, context: str, question: str) -> str: ...

    def stream_answer(self, context: str, question: str) -> Iterator[str]:
        """Yield the answer in pieces. Defaults to one piece."""
        yield self.answer_question(context, question)
