"""Google Gemini models via the ``google-genai`` SDK.

* Embeddings are batched (up to 100 texts per request) and use retrieval task types
  (``RETRIEVAL_DOCUMENT`` for stored text, ``RETRIEVAL_QUERY`` for queries).
* Calls are spread over every configured key (``GEMINI_API_KEY``, ``GEMINI_API_KEY2``, ...)
  with per-key rate limits, failover on 429 and retries on 5xx (see :mod:`.keypool`).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from ..settings import get_settings
from .base import BaseEmbeddingModel, BaseQAModel, BaseSummarizationModel
from .keypool import KeyPool, shared_pool

logger = logging.getLogger(__name__)

DEFAULT_LLM = "gemini-3.5-flash-lite"
DEFAULT_EMBEDDING = "gemini-embedding-001"
EMBED_BATCH_LIMIT = 100

SUMMARY_PROMPT = (
    "Write an objective, factual summary of the following text, including the key "
    "characters, entities, events and details, in under {max_tokens} words.\n\n{context}"
)
QA_PROMPT = (
    "You are a question answering assistant.\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Give a clear, direct and complete answer based on the context."
)


def _make_client(api_key: str) -> Any:
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise ImportError("Gemini support needs `pip install raptor-code[gemini]`") from exc
    return genai.Client(api_key=api_key)


def _pool(api_key: str | None) -> KeyPool:
    keys = [api_key] if api_key else get_settings().require_gemini_keys()
    return shared_pool(keys, _make_client)


def _strip_prefix(name: str) -> str:
    return name.removeprefix("models/")


def _safety_settings() -> list[Any]:
    from google.genai import types

    cats = [
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    ]
    # Benchmark stories include violence and crime; default filters block legitimate summaries.
    return [
        types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.BLOCK_NONE) for c in cats
    ]


class GeminiEmbeddingModel(BaseEmbeddingModel):
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING,
        api_key: str | None = None,
        *,
        document_task_type: str | None = "RETRIEVAL_DOCUMENT",
        query_task_type: str | None = "RETRIEVAL_QUERY",
        output_dimensionality: int | None = None,
        batch_size: int = EMBED_BATCH_LIMIT,
        rpm: int | None = None,
        rate_limit_delay: float | None = None,  # accepted for backward compatibility; unused
    ) -> None:
        self.model_name = _strip_prefix(model_name)
        self.document_task_type = document_task_type
        self.query_task_type = query_task_type
        self.output_dimensionality = output_dimensionality
        self.batch_size = max(1, min(batch_size, EMBED_BATCH_LIMIT))
        self._pool = _pool(api_key)
        self._bucket = f"embed:{self.model_name}"
        self._rpm = rpm if rpm is not None else get_settings().gemini_embed_rpm
        self.model_id = (
            f"gemini:{self.model_name}:doc={document_task_type}:q={query_task_type}"
            f":dim={output_dimensionality}"
        )

    def _config(self, task_type: str | None):
        from google.genai import types

        return types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=self.output_dimensionality
        )

    def _embed_batch(self, texts: list[str], task_type: str | None) -> list[list[float]]:
        config = self._config(task_type)
        resp = self._pool.call(
            self._bucket,
            self._rpm,
            lambda c: c.models.embed_content(model=self.model_name, contents=texts, config=config),
        )
        return [list(e.values) for e in resp.embeddings]

    def _embed(self, texts: Sequence[str], task_type: str | None) -> np.ndarray:
        clean = [(t.replace("\n", " ").strip() or " ") for t in texts]
        out: list[list[float]] = []
        for i in range(0, len(clean), self.batch_size):
            out.extend(self._embed_batch(clean[i : i + self.batch_size], task_type))
        return np.asarray(out, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts, self.document_task_type)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], self.query_task_type)[0]


class _GeminiLLM:
    def __init__(self, model_name: str, api_key: str | None, rpm: int | None) -> None:
        self.model_name = _strip_prefix(model_name)
        self._pool = _pool(api_key)
        self._bucket = f"llm:{self.model_name}"
        self._rpm = rpm if rpm is not None else get_settings().gemini_rpm

    def _generate(self, prompt: str, **config: Any):
        from google.genai import types

        cfg = types.GenerateContentConfig(safety_settings=_safety_settings(), **config)
        return self._pool.call(
            self._bucket,
            self._rpm,
            lambda c: c.models.generate_content(model=self.model_name, contents=prompt, config=cfg),
        )

    @staticmethod
    def _text(resp: Any) -> str | None:
        try:
            text = resp.text
        except (ValueError, AttributeError):
            text = None
        return text.strip() if text else None


class GeminiSummarizationModel(BaseSummarizationModel, _GeminiLLM):
    def __init__(
        self,
        model_name: str = DEFAULT_LLM,
        api_key: str | None = None,
        *,
        prompt_template: str = SUMMARY_PROMPT,
        temperature: float = 0.2,
        rpm: int | None = None,
    ) -> None:
        _GeminiLLM.__init__(self, model_name, api_key, rpm)
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.model_id = f"gemini:{self.model_name}:t={temperature}:p={hash_text(prompt_template)}"

    def summarize(self, context: str, max_tokens: int = 150) -> str:
        prompt = self.prompt_template.format(context=context, max_tokens=max_tokens)
        resp = self._generate(
            prompt, temperature=self.temperature, max_output_tokens=max_tokens * 2
        )
        text = self._text(resp)
        if text:
            return text
        logger.warning(
            "Gemini returned no summary text (likely filtered); using extractive fallback"
        )
        return context[:400].rstrip() + "..."


class GeminiQAModel(BaseQAModel, _GeminiLLM):
    def __init__(
        self,
        model_name: str = DEFAULT_LLM,
        api_key: str | None = None,
        *,
        prompt_template: str = QA_PROMPT,
        temperature: float = 0.0,
        rpm: int | None = None,
    ) -> None:
        _GeminiLLM.__init__(self, model_name, api_key, rpm)
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.model_id = f"gemini:{self.model_name}:t={temperature}:p={hash_text(prompt_template)}"

    def answer_question(self, context: str, question: str) -> str:
        resp = self._generate(
            self.prompt_template.format(context=context, question=question),
            temperature=self.temperature,
        )
        return (
            self._text(resp)
            or "The model returned no answer (the response may have been filtered)."
        )

    def stream_answer(self, context: str, question: str) -> Iterator[str]:
        from google.genai import types

        prompt = self.prompt_template.format(context=context, question=question)
        cfg = types.GenerateContentConfig(
            temperature=self.temperature, safety_settings=_safety_settings()
        )

        def start(client: Any):
            # Errors surface on the first chunk, so pull it inside the retried call.
            it = iter(
                client.models.generate_content_stream(
                    model=self.model_name, contents=prompt, config=cfg
                )
            )
            return next(it, None), it

        first, rest = self._pool.call(self._bucket, self._rpm, start)
        for chunk in _chain(first, rest):
            if self._text(chunk):
                yield chunk.text


def _chain(first: Any, rest: Iterator[Any]) -> Iterator[Any]:
    if first is not None:
        yield first
    yield from rest


def hash_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()[:8]
