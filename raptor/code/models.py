"""Default models for code indexes.

Code summaries use prompts from :mod:`raptor.code.prompts` that are complete
instructions, so the summarizer must send its ``context`` to the LLM unchanged.
"""

from __future__ import annotations

from typing import Any

from ..models import CachedEmbeddingModel, CachedQAModel, CachedSummarizationModel, open_cache
from ..models.base import BaseEmbeddingModel, BaseQAModel, BaseSummarizationModel
from ..settings import Settings, get_settings
from .config import CodeModelsConfig

PASSTHROUGH_TEMPLATE = "{context}"
QA_TEMPLATE = "{context}\n\nQuestion:\n{question}\n\nAnswer:"


class _OllamaPassthrough(BaseSummarizationModel):
    """Ollama summarizer that sends the prompt as-is (the built-in one wraps it)."""

    def __init__(self, model_name: str) -> None:
        from ..models.ollama import OllamaSummarizationModel

        self._inner = OllamaSummarizationModel(model_name)
        self.model_id = f"ollama:{model_name}:passthrough"

    def summarize(self, context: str, max_tokens: int = 150) -> str:
        return self._inner._generate(context, temperature=0.2, num_predict=max_tokens * 2)


class _OllamaCodeQA(BaseQAModel):
    def __init__(self, model_name: str) -> None:
        from ..models.ollama import OllamaQAModel

        self._inner = OllamaQAModel(model_name)
        self.model_id = f"ollama:{model_name}:code-qa"

    def answer_question(self, context: str, question: str) -> str:
        prompt = QA_TEMPLATE.format(context=context, question=question)
        return self._inner._generate(prompt, temperature=0.0)


def build_embedding_model(cfg: CodeModelsConfig) -> BaseEmbeddingModel:
    provider = cfg.embedding_provider or cfg.provider
    if provider == "gemini":
        from ..models.gemini import GeminiEmbeddingModel

        task = cfg.use_task_types
        return GeminiEmbeddingModel(
            cfg.embedding_model,
            document_task_type=cfg.document_task_type if task else None,
            query_task_type=cfg.query_task_type if task else None,
        )
    if provider == "local":
        from ..models.local import SBertEmbeddingModel

        return SBertEmbeddingModel(cfg.embedding_model)
    from ..models.ollama import OllamaEmbeddingModel

    return OllamaEmbeddingModel(cfg.embedding_model)


def build_summarization_model(cfg: CodeModelsConfig) -> BaseSummarizationModel:
    provider = cfg.summary_provider or ("ollama" if cfg.provider == "ollama" else "gemini")
    model = cfg.summary_model or cfg.llm_model
    if provider == "ollama":
        return _OllamaPassthrough(model)
    from ..models.gemini import GeminiSummarizationModel

    return GeminiSummarizationModel(model, prompt_template=PASSTHROUGH_TEMPLATE)


def build_qa_model(cfg: CodeModelsConfig) -> BaseQAModel:
    if cfg.provider == "ollama":
        return _OllamaCodeQA(cfg.llm_model)
    from ..models.gemini import GeminiQAModel

    return GeminiQAModel(cfg.llm_model, prompt_template=QA_TEMPLATE)


class LazyModels:
    """Creates the default models on first use (so read-only commands need no API key)."""

    def __init__(
        self,
        cfg: CodeModelsConfig,
        *,
        embedding: BaseEmbeddingModel | None = None,
        summarization: BaseSummarizationModel | None = None,
        qa: BaseQAModel | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.cfg = cfg
        self._embedding = embedding
        self._summarization = summarization
        self._qa = qa
        self._settings = settings

    def _cached(self, model: Any, kind: str) -> Any:
        settings = self._settings or get_settings()
        if not (self.cfg.cache and settings.cache_enabled):
            return model
        cache = open_cache(settings.cache_dir)
        wrapper = {
            "embedding": CachedEmbeddingModel,
            "summarization": CachedSummarizationModel,
            "qa": CachedQAModel,
        }[kind]
        return wrapper(model, cache)

    @property
    def embedding(self) -> BaseEmbeddingModel:
        if self._embedding is None:
            self._embedding = self._cached(build_embedding_model(self.cfg), "embedding")
        return self._embedding

    @property
    def summarization(self) -> BaseSummarizationModel:
        if self._summarization is None:
            self._summarization = self._cached(build_summarization_model(self.cfg), "summarization")
        return self._summarization

    @property
    def qa(self) -> BaseQAModel:
        if self._qa is None:
            self._qa = self._cached(build_qa_model(self.cfg), "qa")
        return self._qa
