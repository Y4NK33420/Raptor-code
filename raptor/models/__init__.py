"""Model interfaces, providers and the disk cache."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import ModelsConfig
from ..settings import Settings, get_settings
from .base import BaseEmbeddingModel, BaseQAModel, BaseSummarizationModel
from .cache import (
    CachedEmbeddingModel,
    CachedQAModel,
    CachedSummarizationModel,
    DiskCache,
    open_cache,
)

__all__ = [
    "BaseEmbeddingModel",
    "BaseQAModel",
    "BaseSummarizationModel",
    "CachedEmbeddingModel",
    "CachedQAModel",
    "CachedSummarizationModel",
    "DiskCache",
    "ModelBundle",
    "build_models",
    "open_cache",
    "with_cache",
]


@dataclass
class ModelBundle:
    embedding: BaseEmbeddingModel
    summarization: BaseSummarizationModel
    qa: BaseQAModel


def with_cache(bundle: ModelBundle, cache: DiskCache) -> ModelBundle:
    """Wrap every model in ``bundle`` with the disk cache (idempotent)."""
    emb = bundle.embedding
    summ = bundle.summarization
    qa = bundle.qa
    return ModelBundle(
        embedding=emb
        if isinstance(emb, CachedEmbeddingModel)
        else CachedEmbeddingModel(emb, cache),
        summarization=summ
        if isinstance(summ, CachedSummarizationModel)
        else CachedSummarizationModel(summ, cache),
        qa=qa if isinstance(qa, CachedQAModel) else CachedQAModel(qa, cache),
    )


def build_models(cfg: ModelsConfig | None = None, settings: Settings | None = None) -> ModelBundle:
    """Create the embedding, summarization and QA models described by ``cfg``."""
    cfg = cfg or ModelsConfig()
    settings = settings or get_settings()
    emb_provider = cfg.embedding_provider or cfg.provider

    embedding: BaseEmbeddingModel
    if emb_provider == "gemini":
        from .gemini import GeminiEmbeddingModel

        task = cfg.use_task_types
        embedding = GeminiEmbeddingModel(
            cfg.embedding_model,
            document_task_type="RETRIEVAL_DOCUMENT" if task else None,
            query_task_type="RETRIEVAL_QUERY" if task else None,
        )
    elif emb_provider == "local":
        from .local import SBertEmbeddingModel, resolve_local_embedding

        hub_id, q_prefix, d_prefix = resolve_local_embedding(cfg.embedding_model)
        embedding = SBertEmbeddingModel(hub_id, query_prefix=q_prefix, document_prefix=d_prefix)
    else:
        from .ollama import OllamaEmbeddingModel

        embedding = OllamaEmbeddingModel(cfg.embedding_model)

    summarization: BaseSummarizationModel
    qa: BaseQAModel
    if cfg.provider == "ollama":
        from .ollama import OllamaQAModel, OllamaSummarizationModel

        summarization, qa = OllamaSummarizationModel(cfg.llm_model), OllamaQAModel(cfg.llm_model)
    else:
        # "local" means local embeddings; generation still uses Gemini unless provider=ollama.
        from .gemini import GeminiQAModel, GeminiSummarizationModel

        summarization, qa = GeminiSummarizationModel(cfg.llm_model), GeminiQAModel(cfg.llm_model)

    bundle = ModelBundle(embedding, summarization, qa)
    if cfg.cache and settings.cache_enabled:
        bundle = with_cache(bundle, open_cache(settings.cache_dir))
    return bundle
