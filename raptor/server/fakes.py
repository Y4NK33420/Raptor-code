"""Deterministic offline models for demos, UI development and end-to-end tests.

They make no network calls and always give the same output for the same input, so a
Playwright run or a screen recording is reproducible and costs no API quota.
"""

from __future__ import annotations

import hashlib
import itertools
import re
import time
from collections.abc import Iterator, Sequence

import numpy as np

from ..models import ModelBundle
from ..models.base import BaseEmbeddingModel, BaseQAModel, BaseSummarizationModel

FAKE_DIM = 256
_WORD = re.compile(r"[a-z][a-z']+")
_STOP = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "as",
        "is",
        "was",
        "were",
        "be",
        "been",
        "are",
        "it",
        "its",
        "this",
        "that",
        "he",
        "she",
        "they",
        "them",
        "his",
        "her",
        "their",
        "we",
        "you",
        "i",
        "not",
        "but",
        "so",
        "if",
        "then",
        "than",
        "there",
        "here",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "some",
        "no",
        "do",
        "did",
        "does",
        "had",
        "has",
        "have",
        "into",
        "out",
        "up",
        "down",
        "over",
    ]
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_OPTION = re.compile(r"^\s*(\d+)\.\s+(.*)$", re.M)


def _words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


def _bucket(token: str) -> tuple[int, float]:
    h = int(hashlib.md5(token.encode()).hexdigest(), 16)
    return h % FAKE_DIM, 1.0 if (h >> 64) & 1 else -1.0


class FakeEmbeddingModel(BaseEmbeddingModel):
    """Signed feature hashing of content words and bigrams: shared vocabulary → similar vectors."""

    model_id = "fake:hash-embed-256"

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(FAKE_DIM, dtype=np.float32)
        words = _words(text)
        for w in words:
            i, s = _bucket(w)
            v[i] += s
        for a, b in itertools.pairwise(words):
            i, s = _bucket(f"{a}_{b}")
            v[i] += 0.5 * s
        n = float(np.linalg.norm(v))
        if n == 0:
            v[_bucket(text)[0]] = 1.0
            return v
        return v / n

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, FAKE_DIM), dtype=np.float32)
        return np.stack([self._vector(t) for t in texts])


class FakeSummarizationModel(BaseSummarizationModel):
    """Extractive summary: the most central sentences of the input, in original order."""

    model_id = "fake:extractive-summary"

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    def summarize(self, context: str, max_tokens: int = 150) -> str:
        if self.delay:
            time.sleep(self.delay)
        sentences = [s.strip() for s in _SENTENCE.split(" ".join(context.split())) if s.strip()]
        if not sentences:
            return context[:200]
        freq: dict[str, int] = {}
        for w in _words(context):
            freq[w] = freq.get(w, 0) + 1

        def score(s: str) -> float:
            ws = _words(s)
            return sum(freq.get(w, 0) for w in ws) / (len(ws) + 3)

        ranked = sorted(range(len(sentences)), key=lambda i: (-score(sentences[i]), i))
        picked: list[int] = []
        budget = max(12, int(max_tokens * 0.75))
        used = 0
        for i in ranked:
            n = len(sentences[i].split())
            if picked and used + n > budget:
                continue
            picked.append(i)
            used += n
            if used >= budget:
                break
        text = " ".join(sentences[i] for i in sorted(picked))
        words = text.split()
        return " ".join(words[:budget]) + ("…" if len(words) > budget else "")


class FakeQAModel(BaseQAModel):
    """Answers from the context by sentence overlap; picks MCQ options by overlap too."""

    model_id = "fake:extractive-qa"

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    def answer_question(self, context: str, question: str) -> str:
        return "".join(self._compose(context, question))

    def stream_answer(self, context: str, question: str) -> Iterator[str]:
        for piece in self._compose(context, question):
            if self.delay:
                time.sleep(self.delay)
            yield piece

    # ---------------------------------------------------------------- internals
    def _compose(self, context: str, question: str) -> list[str]:
        q_part = question.split("Options:")[0]
        options = [m.group(2).strip() for m in _OPTION.finditer(question.split("Options:")[-1])]
        ctx_words = set(_words(context))
        evidence = self._best_sentences(context, q_part, 2)
        if "Options:" in question and options:
            scores = [len(set(_words(o)) & ctx_words) / (len(set(_words(o))) + 1) for o in options]
            best = int(np.argmax(scores)) + 1
            text = f"Option {best}. The retrieved passages support this choice"
            text += f': "{evidence[0]}"' if evidence else "."
            if len(evidence) > 1:
                text += f' They also note: "{evidence[1]}"'
        elif evidence:
            text = " ".join(evidence)
        else:
            text = "The retrieved context does not contain enough information to answer."
        # Stream in word-sized pieces like a real model.
        return [w + " " for w in text.split()]

    @staticmethod
    def _best_sentences(context: str, question: str, n: int) -> list[str]:
        q = set(_words(question))
        sentences = [s.strip() for s in _SENTENCE.split(" ".join(context.split())) if s.strip()]
        scored = sorted(
            ((len(q & set(_words(s))), -i, s) for i, s in enumerate(sentences)), reverse=True
        )
        out = [s for score, _, s in scored[:n] if score > 0]
        return [s if len(s) < 280 else s[:277] + "…" for s in out]


def fake_bundle(delay: float = 0.0) -> ModelBundle:
    return ModelBundle(
        embedding=FakeEmbeddingModel(),
        summarization=FakeSummarizationModel(delay=delay * 4),
        qa=FakeQAModel(delay=delay),
    )
