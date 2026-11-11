"""Declarative configuration for building and querying a RAPTOR index.

Every experiment is a :class:`RaptorConfig`, which round-trips through YAML so runs
are reproducible and comparable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChunkingConfig(_Section):
    strategy: str = "sentence"
    max_tokens: int = Field(100, ge=1)
    overlap_tokens: int = Field(0, ge=0)
    params: dict[str, Any] = Field(default_factory=dict)


class ClusteringConfig(_Section):
    strategy: str = "raptor_gmm"
    reduction_dimension: int = Field(10, ge=1)
    threshold: float = Field(0.1, ge=0.0, le=1.0)
    max_cluster_tokens: int = Field(3500, ge=1)
    random_state: int = 224
    params: dict[str, Any] = Field(default_factory=dict)


class BuilderConfig(_Section):
    max_layers: int = Field(5, ge=1)
    summarization_length: int = Field(100, ge=1)
    # Stop adding layers once a layer has this many nodes or fewer.
    # ``None`` means ``reduction_dimension + 1`` (the original RAPTOR rule).
    min_nodes_to_cluster: int | None = Field(None, ge=1)
    max_workers: int = Field(4, ge=1)
    embedding_key: str = "EMB"


class RetrievalConfig(_Section):
    mode: Literal["collapsed", "traversal"] = "collapsed"
    top_k: int = Field(10, ge=1)
    max_tokens: int = Field(3500, ge=1)
    selection_mode: Literal["top_k", "threshold"] = "top_k"
    # Minimum cosine similarity for ``selection_mode="threshold"``.
    threshold: float = Field(0.5, ge=-1.0, le=1.0)
    start_layer: int | None = Field(None, ge=0)
    num_layers: int | None = Field(None, ge=1)


class ModelsConfig(_Section):
    provider: Literal["gemini", "local", "ollama"] = "gemini"
    embedding_model: str = "gemini-embedding-001"
    llm_model: str = "gemini-3.5-flash-lite"
    embedding_provider: Literal["gemini", "local", "ollama"] | None = None
    use_task_types: bool = True
    cache: bool = True


class RaptorConfig(_Section):
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    builder: BuilderConfig = Field(default_factory=BuilderConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)

    # ----------------------------------------------------------------- YAML I/O
    @classmethod
    def from_yaml(cls, path: str | Path) -> RaptorConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path | None = None) -> str:
        text = yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def fingerprint(self, *, sections: tuple[str, ...] | None = None) -> str:
        """Stable short hash of the config (or of selected sections)."""
        data = self.model_dump(mode="json")
        if sections is not None:
            data = {k: data[k] for k in sections}
        blob = json.dumps(data, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    @property
    def stop_threshold(self) -> int:
        if self.builder.min_nodes_to_cluster is not None:
            return self.builder.min_nodes_to_cluster
        return self.clustering.reduction_dimension + 1
