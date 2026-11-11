"""RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.

Based on Sarthi et al., "RAPTOR: Recursive Abstractive Processing for Tree-Organized
Retrieval" (ICLR 2024) and the original implementation at
https://github.com/parthsarthi03/raptor.
"""

from __future__ import annotations

import logging
from typing import Any

from .builder import TreeBuilder
from .chunking import Chunk, Chunker, SentenceChunker, split_text
from .clustering import ClusteringAlgorithm, RaptorGMMClustering
from .config import (
    BuilderConfig,
    ChunkingConfig,
    ClusteringConfig,
    ModelsConfig,
    RaptorConfig,
    RetrievalConfig,
)
from .models import ModelBundle, build_models
from .models.base import BaseEmbeddingModel, BaseQAModel, BaseSummarizationModel
from .pipeline import RetrievalAugmentation, RetrievalAugmentationConfig
from .retrieval import FlatRAG, RetrievalResult, RetrievedNode, TreeRetriever
from .storage import load_legacy_pickle, load_tree, save_tree
from .types import Node, Tree

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.2.0"

__all__ = [
    "BaseEmbeddingModel",
    "BaseQAModel",
    "BaseSummarizationModel",
    "BuilderConfig",
    "Chunk",
    "Chunker",
    "ChunkingConfig",
    "ClusteringAlgorithm",
    "ClusteringConfig",
    "FlatRAG",
    "GeminiEmbeddingModel",
    "GeminiQAModel",
    "GeminiSummarizationModel",
    "ModelBundle",
    "ModelsConfig",
    "Node",
    "RaptorConfig",
    "RaptorGMMClustering",
    "RetrievalAugmentation",
    "RetrievalAugmentationConfig",
    "RetrievalConfig",
    "RetrievalResult",
    "RetrievedNode",
    "SBertEmbeddingModel",
    "SentenceChunker",
    "Tree",
    "TreeBuilder",
    "TreeRetriever",
    "build_models",
    "load_legacy_pickle",
    "load_tree",
    "save_tree",
    "split_text",
]

_LAZY = {
    # Provider classes import optional SDKs, so load them on first access.
    "GeminiEmbeddingModel": ("raptor.models.gemini", "GeminiEmbeddingModel"),
    "GeminiQAModel": ("raptor.models.gemini", "GeminiQAModel"),
    "GeminiSummarizationModel": ("raptor.models.gemini", "GeminiSummarizationModel"),
    "SBertEmbeddingModel": ("raptor.models.local", "SBertEmbeddingModel"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module 'raptor' has no attribute {name!r}")
