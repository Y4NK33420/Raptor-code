"""Ollama-served local models (optional). Useful to avoid API quotas while developing."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from ..settings import get_settings
from .base import BaseEmbeddingModel, BaseQAModel, BaseSummarizationModel
from .gemini import QA_PROMPT, SUMMARY_PROMPT


def _http():
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise ImportError("Ollama support needs `pip install raptor-code[ollama]`") from exc
    return httpx


class _Ollama:
    def __init__(self, model_name: str, host: str | None, timeout: float) -> None:
        self.model_name = model_name
        self.host = (host or get_settings().ollama_host).rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        httpx = _http()
        resp = httpx.post(f"{self.host}{path}", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _generate(self, prompt: str, **options: Any) -> str:
        data = self._post(
            "/api/generate",
            {"model": self.model_name, "prompt": prompt, "stream": False, "options": options},
        )
        return str(data.get("response", "")).strip()


class OllamaEmbeddingModel(BaseEmbeddingModel, _Ollama):
    def __init__(
        self, model_name: str = "nomic-embed-text", host: str | None = None, timeout: float = 120
    ) -> None:
        _Ollama.__init__(self, model_name, host, timeout)
        self.model_id = f"ollama:{model_name}"

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        data = self._post("/api/embed", {"model": self.model_name, "input": list(texts)})
        return np.asarray(data["embeddings"], dtype=np.float32)


class OllamaSummarizationModel(BaseSummarizationModel, _Ollama):
    def __init__(
        self, model_name: str = "qwen2.5:3b", host: str | None = None, timeout: float = 300
    ) -> None:
        _Ollama.__init__(self, model_name, host, timeout)
        self.model_id = f"ollama:{model_name}"

    def summarize(self, context: str, max_tokens: int = 150) -> str:
        prompt = SUMMARY_PROMPT.format(context=context, max_tokens=max_tokens)
        return self._generate(prompt, temperature=0.2, num_predict=max_tokens * 2)


class OllamaQAModel(BaseQAModel, _Ollama):
    def __init__(
        self, model_name: str = "qwen2.5:3b", host: str | None = None, timeout: float = 300
    ) -> None:
        _Ollama.__init__(self, model_name, host, timeout)
        self.model_id = f"ollama:{model_name}"

    def answer_question(self, context: str, question: str) -> str:
        return self._generate(QA_PROMPT.format(context=context, question=question), temperature=0.0)

    def stream_answer(self, context: str, question: str) -> Iterator[str]:
        yield self.answer_question(context, question)
