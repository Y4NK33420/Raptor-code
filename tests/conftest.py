"""Shared fixtures: deterministic offline models and a synthetic multi-topic document."""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Sequence

import numpy as np
import pytest

from raptor.models.base import BaseEmbeddingModel, BaseQAModel, BaseSummarizationModel

DIM = 64
_WORD = re.compile(r"[a-z]+")


class HashEmbedder(BaseEmbeddingModel):
    """Bag-of-words hashed into DIM buckets: texts sharing words get similar vectors."""

    model_id = "test:hash-bow"

    def __init__(self) -> None:
        self.doc_calls = 0
        self.texts_embedded = 0

    @staticmethod
    def vector(text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for w in _WORD.findall(text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v + 1e-3

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.doc_calls += 1
        self.texts_embedded += len(texts)
        return (
            np.stack([self.vector(t) for t in texts]) if texts else np.zeros((0, DIM), np.float32)
        )


class EchoSummarizer(BaseSummarizationModel):
    """'Summarises' by keeping the first sentence of each paragraph."""

    model_id = "test:echo"

    def __init__(self) -> None:
        self.calls = 0

    def summarize(self, context: str, max_tokens: int = 150) -> str:
        self.calls += 1
        firsts = [p.split(". ")[0].strip() for p in context.split("\n\n") if p.strip()]
        words = " ".join(firsts).split()
        return " ".join(words[:max_tokens])


class EchoQA(BaseQAModel):
    model_id = "test:echo-qa"

    def answer_question(self, context: str, question: str) -> str:
        return f"Q={question} | ctx_chars={len(context)}"


TOPICS = {
    "ocean": "ocean waves tide coral reef fish sailor ship harbor salt",
    "forest": "forest trees moss deer owl pine trail ranger campfire leaves",
    "space": "rocket orbit planet astronaut star galaxy launch comet moon",
    "kitchen": "kitchen recipe flour oven baker bread butter spice dough",
}


def make_document(sentences_per_topic: int = 30, seed: int = 0) -> str:
    """Paragraphs that each stick to one topic, so clustering has real structure."""
    rng = random.Random(seed)
    paragraphs = []
    for name, vocab in TOPICS.items():
        words = vocab.split()
        sents = []
        for _ in range(sentences_per_topic):
            picked = rng.sample(words, 5)
            sents.append(
                f"The {picked[0]} and the {picked[1]} met {picked[2]} near {picked[3]} by {picked[4]}."
            )
        paragraphs.append(f"{name.title()} chapter. " + " ".join(sents))
    return "\n\n".join(paragraphs)


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture
def summarizer() -> EchoSummarizer:
    return EchoSummarizer()


@pytest.fixture
def qa() -> EchoQA:
    return EchoQA()


@pytest.fixture(scope="session")
def document() -> str:
    return make_document()
