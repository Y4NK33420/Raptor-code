"""Configuration for code indexes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..config import ClusteringConfig, ModelsConfig

SummaryDepth = Literal["symbol", "file"]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IngestConfig(_Section):
    languages: list[str] = Field(default_factory=lambda: ["python"])
    #: Extra gitignore-style patterns to skip (on top of .gitignore and the built-in list).
    exclude: list[str] = Field(default_factory=list)
    #: If set, only paths matching one of these gitignore-style patterns are indexed.
    include: list[str] = Field(default_factory=list)
    max_file_bytes: int = Field(1_000_000, ge=1)
    use_git: bool = True


class ChunkConfig(_Section):
    #: Symbols longer than this are split into parts.
    max_symbol_tokens: int = Field(400, ge=32)
    #: Adjacent sibling symbols shorter than this are merged into one leaf.
    min_symbol_tokens: int = Field(40, ge=0)
    #: Add a ``path › Class › def sig`` header and the imports the symbol uses.
    contextual_header: bool = True


class SummaryConfig(_Section):
    #: ``symbol``: summarize every symbol, then classes and files from those summaries.
    #: ``file``: skip per-symbol summaries; files are summarized from their source.
    depth: SummaryDepth = "file"
    #: Target summary length in words.
    length: int = Field(160, ge=16)
    #: Token budget for the source excerpt in a file prompt (file depth).
    file_context_tokens: int = Field(3000, ge=200)
    #: Token budget for child summaries in a directory / repo / concept prompt.
    group_context_tokens: int = Field(4000, ge=200)
    max_workers: int = Field(4, ge=1)


class ConceptConfig(_Section):
    enabled: bool = True
    clustering: ClusteringConfig = Field(
        default_factory=lambda: ClusteringConfig(reduction_dimension=8, max_cluster_tokens=6000)
    )
    #: Clusters with fewer members than this do not become concept nodes.
    min_members: int = Field(2, ge=1)
    #: Skip the concept layer for repos with fewer file+class nodes than this.
    min_nodes: int = Field(6, ge=2)


class CodeRetrievalConfig(_Section):
    dense: bool = True
    bm25: bool = True
    graph_expansion: bool = True
    #: Descend from matching summary nodes into their best-matching leaves.
    descend: bool = True
    routing: bool = True
    rrf_k: int = Field(60, ge=1)
    #: Candidates taken from each ranker before fusion.
    candidates: int = Field(100, ge=1)
    #: Score multiplier for nodes added by graph expansion / descent (relative to the hit).
    expansion_decay: float = Field(0.85, ge=0.0, le=1.0)
    #: Neighbours added per hit by graph expansion.
    expansion_per_hit: int = Field(3, ge=0)
    #: Token budget for the context given to the reader in ``ask``.
    max_context_tokens: int = Field(6000, ge=100)
    #: Leave out node kinds entirely, e.g. ``["concept"]`` or ``["class", "file", "dir", "repo", "concept"]``.
    exclude_kinds: list[str] = Field(default_factory=list)


class CodeModelsConfig(ModelsConfig):
    #: Query task type for code search ("CODE_RETRIEVAL_QUERY" for Gemini).
    query_task_type: str | None = "CODE_RETRIEVAL_QUERY"
    document_task_type: str | None = "RETRIEVAL_DOCUMENT"
    #: Override the provider / model used for summaries only (e.g. "ollama").
    summary_provider: Literal["gemini", "ollama"] | None = None
    summary_model: str | None = None


class CodeConfig(_Section):
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    chunking: ChunkConfig = Field(default_factory=ChunkConfig)
    summary: SummaryConfig = Field(default_factory=SummaryConfig)
    concepts: ConceptConfig = Field(default_factory=ConceptConfig)
    retrieval: CodeRetrievalConfig = Field(default_factory=CodeRetrievalConfig)
    models: CodeModelsConfig = Field(default_factory=CodeModelsConfig)
    embedding_key: str = "EMB"

    @classmethod
    def from_yaml(cls, path: str | Path) -> CodeConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path | None = None) -> str:
        text = yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def fingerprint(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]
