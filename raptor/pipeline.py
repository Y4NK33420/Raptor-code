"""High-level API: build an index, retrieve, and answer questions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .builder import ProgressCallback, TreeBuilder
from .chunking import Chunk
from .config import RaptorConfig
from .models import ModelBundle, build_models, open_cache, with_cache
from .models.base import BaseEmbeddingModel, BaseQAModel, BaseSummarizationModel
from .retrieval import RetrievalResult, TreeRetriever
from .settings import get_settings
from .storage import load_tree, save_tree
from .types import Tree

logger = logging.getLogger(__name__)


class RetrievalAugmentationConfig:
    """Backward-compatible configuration matching the original RAPTOR keyword arguments.

    New code should use :class:`raptor.RaptorConfig` directly. This class maps the old
    ``tb_*`` / ``tr_*`` arguments onto it and carries any model instances passed in.

    Note: ``tr_threshold`` is now a minimum cosine *similarity* (the original compared
    distances the wrong way round).
    """

    _IGNORED: frozenset[str] = frozenset(
        {
            "tree_builder_type",
            "tr_tokenizer",
            "tb_tokenizer",
            "tr_context_embedding_model",
            "tb_cluster_embedding_model",
            "tb_threshold",
            "tb_top_k",
            "tb_selection_mode",
        }
    )

    def __init__(
        self,
        qa_model: BaseQAModel | None = None,
        embedding_model: BaseEmbeddingModel | None = None,
        summarization_model: BaseSummarizationModel | None = None,
        *,
        tb_max_tokens: int = 100,
        tb_num_layers: int = 5,
        tb_summarization_length: int = 100,
        tb_summarization_model: BaseSummarizationModel | None = None,
        tb_embedding_models: dict[str, BaseEmbeddingModel] | None = None,
        tr_top_k: int = 10,
        tr_threshold: float = 0.5,
        tr_selection_mode: str = "top_k",
        tr_embedding_model: BaseEmbeddingModel | None = None,
        tr_num_layers: int | None = None,
        tr_start_layer: int | None = None,
        raptor_config: RaptorConfig | None = None,
        **ignored: Any,
    ) -> None:
        unknown = set(ignored) - self._IGNORED
        if unknown:
            raise TypeError(f"unexpected arguments: {sorted(unknown)}")
        if "tree_builder_config" in ignored or "tree_retriever_config" in ignored:
            raise TypeError(
                "tree_builder_config/tree_retriever_config are no longer supported; use RaptorConfig"
            )

        embedding_key = "EMB"
        if tb_embedding_models:
            if embedding_model is not None:
                raise ValueError("pass only one of embedding_model / tb_embedding_models")
            embedding_key, embedding_model = next(iter(tb_embedding_models.items()))
        if summarization_model is not None and tb_summarization_model is not None:
            raise ValueError("pass only one of summarization_model / tb_summarization_model")

        cfg = raptor_config.model_copy(deep=True) if raptor_config else RaptorConfig()
        if raptor_config is None:
            cfg.chunking.max_tokens = tb_max_tokens
            cfg.builder.max_layers = tb_num_layers
            cfg.builder.summarization_length = tb_summarization_length
            cfg.builder.embedding_key = embedding_key
            cfg.retrieval.top_k = tr_top_k
            cfg.retrieval.threshold = tr_threshold
            cfg.retrieval.selection_mode = tr_selection_mode  # type: ignore[assignment]
            cfg.retrieval.num_layers = tr_num_layers
            cfg.retrieval.start_layer = tr_start_layer
        self.raptor = cfg
        self.embedding_model = embedding_model or tr_embedding_model
        self.summarization_model = summarization_model or tb_summarization_model
        self.qa_model = qa_model


class RetrievalAugmentation:
    """Build, query and persist a RAPTOR index.

    >>> ra = RetrievalAugmentation(RaptorConfig())          # models from config (Gemini by default)
    >>> ra.add_documents(open("book.txt").read())
    >>> ra.answer_question("How does the story end?")
    >>> ra.save("indexes/book")
    """

    def __init__(
        self,
        config: RaptorConfig | RetrievalAugmentationConfig | None = None,
        tree: Tree | str | Path | None = None,
        *,
        embedding_model: BaseEmbeddingModel | None = None,
        summarization_model: BaseSummarizationModel | None = None,
        qa_model: BaseQAModel | None = None,
        cache: bool = False,
    ) -> None:
        if isinstance(config, RetrievalAugmentationConfig):
            embedding_model = embedding_model or config.embedding_model
            summarization_model = summarization_model or config.summarization_model
            qa_model = qa_model or config.qa_model
            config = config.raptor
        elif config is not None and not isinstance(config, RaptorConfig):
            raise TypeError("config must be a RaptorConfig or RetrievalAugmentationConfig")
        self.config: RaptorConfig = config or RaptorConfig()
        self._embedding = embedding_model
        self._summarization = summarization_model
        self._qa = qa_model
        self._cache = cache
        self._bundle: ModelBundle | None = None

        if isinstance(tree, (str, Path)):
            tree = load_tree(tree)
        elif tree is not None and not isinstance(tree, Tree):
            raise TypeError("tree must be a Tree, a path to a saved index, or None")
        self.tree: Tree | None = tree
        self._retriever: TreeRetriever | None = None

    # ------------------------------------------------------------------ models
    def _models(self) -> ModelBundle:
        if self._bundle is None:
            need_default = not (self._embedding and self._summarization and self._qa)
            defaults = build_models(self.config.models) if need_default else None
            bundle = ModelBundle(
                embedding=self._embedding or defaults.embedding,  # type: ignore[union-attr]
                summarization=self._summarization or defaults.summarization,  # type: ignore[union-attr]
                qa=self._qa or defaults.qa,  # type: ignore[union-attr]
            )
            settings = get_settings()
            if self._cache and settings.cache_enabled:
                bundle = with_cache(bundle, open_cache(settings.cache_dir))
            self._bundle = bundle
        return self._bundle

    @property
    def embedding_model(self) -> BaseEmbeddingModel:
        return self._embedding if self._embedding and not self._cache else self._models().embedding

    @property
    def summarization_model(self) -> BaseSummarizationModel:
        return self._models().summarization

    @property
    def qa_model(self) -> BaseQAModel:
        return self._qa if self._qa and not self._cache else self._models().qa

    @property
    def retriever(self) -> TreeRetriever:
        if self.tree is None:
            raise ValueError("no index yet: call add_documents() or pass a tree")
        if self._retriever is None or self._retriever.tree is not self.tree:
            self._retriever = TreeRetriever(self.tree, self.embedding_model, self.config.retrieval)
        return self._retriever

    def _builder(self) -> TreeBuilder:
        return TreeBuilder(self.config, self.embedding_model, self.summarization_model)

    # ------------------------------------------------------------------ build
    def add_documents(
        self, docs: str, *, overwrite: bool = False, progress: ProgressCallback | None = None
    ) -> Tree:
        """Build a new index from ``docs``. Refuses to replace an existing one unless ``overwrite``."""
        if self.tree is not None and not overwrite:
            raise ValueError(
                "an index already exists; pass overwrite=True to replace it or use add_to_existing()"
            )
        self.tree = self._builder().build_from_text(docs, progress=progress)
        self._retriever = None
        return self.tree

    def add_to_existing(self, docs: str, *, progress: ProgressCallback | None = None) -> Tree:
        """Add text to the index: keeps existing leaves (and their embeddings), rebuilds upper layers.

        With the disk cache enabled, summaries of unchanged clusters are reused.
        """
        if self.tree is None:
            return self.add_documents(docs, progress=progress)
        import numpy as np

        builder = self._builder()
        key = self.tree.embedding_key
        old = [self.tree.leaf_nodes[i] for i in sorted(self.tree.leaf_nodes)]
        new_chunks = builder.chunker.split(docs)
        chunks = [Chunk(n.text, metadata=dict(n.metadata)) for n in old] + new_chunks
        new_emb = builder.embedding_model.embed_documents([c.text for c in new_chunks])
        old_emb = np.stack([np.asarray(n.embeddings[key], dtype=np.float32) for n in old])
        emb = np.concatenate([old_emb, new_emb]) if len(new_chunks) else old_emb
        self.tree = builder.build_from_chunks(chunks, leaf_embeddings=emb, progress=progress)
        self._retriever = None
        return self.tree

    # --------------------------------------------------------------- retrieve
    def retrieve_nodes(self, question: str, **kwargs: Any) -> RetrievalResult:
        """Retrieve with full detail (nodes, layers, similarity scores)."""
        return self.retriever.retrieve(question, **kwargs)

    def retrieve(
        self,
        question: str,
        start_layer: int | None = None,
        num_layers: int | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        collapse_tree: bool = True,
        return_layer_information: bool = False,
    ):
        """Return the retrieved context (and optionally per-node layer information)."""
        result = self.retrieve_nodes(
            question,
            top_k=top_k,
            max_tokens=max_tokens,
            mode="collapsed" if collapse_tree else "traversal",
            start_layer=start_layer,
            num_layers=num_layers,
        )
        if return_layer_information:
            return result.context, result.layer_information()
        return result.context

    def answer_question(
        self,
        question: str,
        top_k: int | None = None,
        start_layer: int | None = None,
        num_layers: int | None = None,
        max_tokens: int | None = None,
        collapse_tree: bool = True,
        return_layer_information: bool = False,
    ):
        context, layers = self.retrieve(
            question, start_layer, num_layers, top_k, max_tokens, collapse_tree, True
        )
        answer = self.qa_model.answer_question(context, question)
        return (answer, layers) if return_layer_information else answer

    # ----------------------------------------------------------------- persist
    def save(self, path: str | Path) -> Path:
        if self.tree is None:
            raise ValueError("there is no index to save")
        return save_tree(self.tree, path)
