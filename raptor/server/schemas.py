"""Request/response models. They define the OpenAPI schema the frontend's types come from."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow")


# ------------------------------------------------------------------- indexes
class IndexSource(_Open):
    type: str = "text"
    set_id: str | None = None
    title: str | None = None
    filename: str | None = None
    label: str | None = None


class IndexInfo(BaseModel):
    id: str
    name: str
    origin: Literal["demo", "user"]
    kind: str
    num_nodes: int
    num_layers: int
    layer_counts: dict[str, int]
    leaf_chars: int
    embedding_model: str | None = None
    summarization_model: str | None = None
    created_at: str
    source: IndexSource
    preset: str | None = None
    build_seconds: float | None = None


class Preset(BaseModel):
    id: str
    label: str
    description: str
    config: dict[str, Any]


class BuildAccepted(BaseModel):
    job_id: str
    index_id: str


class JobInfo(BaseModel):
    id: str
    kind: str
    status: Literal["queued", "running", "done", "error"]
    meta: dict[str, Any]
    created_at: float
    result: dict[str, Any] | None = None
    error: str | None = None
    n_events: int


class ServerInfo(BaseModel):
    version: str
    fake_models: bool
    code_backend: Literal["raptor.code", "stub"]
    runs_backend: Literal["raptor.eval.runs", "files"]
    quality_available: bool
    llm_model: str
    embedding_model: str


# ---------------------------------------------------------------------- tree
class TreeNode(BaseModel):
    id: int
    layer: int
    preview: str
    children: list[int]
    parents: list[int]
    tokens: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class TreeEdge(BaseModel):
    source: int  # parent
    target: int  # child


class TreeData(BaseModel):
    id: str
    num_layers: int
    layer_counts: dict[str, int]
    nodes: list[TreeNode]
    edges: list[TreeEdge]


class NodeDetail(BaseModel):
    id: int
    layer: int
    text: str
    tokens: int
    children: list[int]
    parents: list[int]
    siblings: list[int]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProjectionPoint(BaseModel):
    id: int
    x: float
    y: float
    layer: int
    parents: list[int]
    cluster: int | None = None


class Projection(BaseModel):
    method: str
    points: list[ProjectionPoint]


# --------------------------------------------------------------------- query
class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(None, ge=1, le=100)
    max_tokens: int | None = Field(None, ge=50, le=20000)
    mode: Literal["collapsed", "traversal"] | None = None


class CompareRequest(BaseModel):
    question: str | None = Field(None, max_length=4000)
    question_id: str | None = None
    set_id: str | None = None
    top_k: int = Field(10, ge=1, le=100)
    max_tokens: int = Field(2000, ge=50, le=20000)


class RetrievedNodeOut(BaseModel):
    id: int
    layer: int
    score: float
    tokens: int
    preview: str


class RetrievalEvent(BaseModel):
    side: Literal["raptor", "flat"] | None = None
    nodes: list[RetrievedNodeOut]
    layer_counts: dict[str, int]
    context_tokens: int
    latency_s: float


class TokenEvent(BaseModel):
    side: Literal["raptor", "flat"] | None = None
    text: str


class AnswerDoneEvent(BaseModel):
    side: Literal["raptor", "flat"] | None = None
    answer: str
    latency_s: float
    predicted: int | None = None
    correct: bool | None = None


class CompareQuestionEvent(BaseModel):
    question: str
    question_id: str | None = None
    set_id: str | None = None
    options: list[str] | None = None
    gold: int | None = None
    difficult: bool | None = None
    budget: dict[str, int]


class BuildSummaryEvent(BaseModel):
    layer: int
    id: int
    preview: str
    cost: dict[str, int]
    t: float


class BuildClustersEvent(BaseModel):
    layer: int
    input: int
    parents: list[dict[str, Any]]
    t: float


class BuildChunksEvent(BaseModel):
    count: int
    chars: int
    chunks: list[dict[str, Any]]
    t: float


class ErrorEvent(BaseModel):
    message: str


class EventCatalog(BaseModel):
    """Payload shapes of the server-sent events (documented here for type generation)."""

    retrieval: RetrievalEvent
    token: TokenEvent
    answer: AnswerDoneEvent
    question: CompareQuestionEvent
    build_chunks: BuildChunksEvent
    build_clusters: BuildClustersEvent
    build_summary: BuildSummaryEvent
    citations: list[CodeHitOut]
    error: ErrorEvent


# -------------------------------------------------------------------- quality
class QualityStory(BaseModel):
    set_id: str
    article_id: str
    title: str
    author: str
    topic: str
    word_count: int
    n_questions: int
    n_hard: int
    index_ids: list[str] = Field(default_factory=list)


class QualityQuestionOut(BaseModel):
    question_id: str
    question: str
    options: list[str]
    gold: int | None
    difficult: bool


class QualityStoryDetail(QualityStory):
    article: str
    questions: list[QualityQuestionOut]


# ----------------------------------------------------------------------- runs
class RunSummary(_Open):
    run_id: str
    name: str | None = None
    created_at: str | None = None
    split: str | None = None
    method: str | None = None
    config_fingerprint: str | None = None
    factors: dict[str, Any] = Field(default_factory=dict)
    n_questions: int = 0
    accuracy: float = 0.0
    accuracy_hard: float | None = None
    ci95: list[float] = Field(default_factory=list)
    layer_share: dict[str, float] = Field(default_factory=dict)
    cost: dict[str, float] = Field(default_factory=dict)
    build_seconds: float | None = None
    query_seconds_mean: float | None = None
    tree_stats: dict[str, float] = Field(default_factory=dict)
    origin: str = "artifact"


class RunResultRow(_Open):
    run_id: str | None = None
    set_id: str
    question_id: str
    question: str | None = None
    options: list[str] | None = None
    gold: int | None = None
    predicted: int | None = None
    correct: bool
    difficult: bool | None = None
    retrieved: list[dict[str, Any]] = Field(default_factory=list)
    context_tokens: int | None = None
    latency_s: float | None = None
    rationale: str | None = None


class RunDetail(BaseModel):
    summary: RunSummary
    config: dict[str, Any]
    config_yaml: str
    results: list[RunResultRow]


class CompareQuestion(BaseModel):
    set_id: str
    question_id: str
    question: str | None = None
    options: list[str] | None = None
    gold: int | None = None
    difficult: bool = False
    a_predicted: int | None = None
    b_predicted: int | None = None
    a_rationale: str | None = None
    b_rationale: str | None = None
    a_layers: dict[str, int] = Field(default_factory=dict)
    b_layers: dict[str, int] = Field(default_factory=dict)
    status: Literal["fixed", "broken", "both_right", "both_wrong"]


class RunCompare(_Open):
    a: str
    b: str
    n_paired: int = 0
    accuracy_a: float = 0.0
    accuracy_b: float = 0.0
    delta: float
    delta_ci95: list[float] = Field(default_factory=list)
    mcnemar_p: float
    fixed: int
    broken: int
    both_right: int
    both_wrong: int
    questions: list[CompareQuestion] = Field(default_factory=list)


# ----------------------------------------------------------------------- code
class CodeRepo(BaseModel):
    id: str
    name: str
    repo_path: str
    n_files: int
    n_symbols: int
    n_nodes: int
    num_layers: int
    stub: bool
    origin: str


class CodeSymbolRef(BaseModel):
    id: int
    symbol: str
    kind: str
    start_line: int
    end_line: int


class CodeFile(BaseModel):
    path: str
    language: str
    symbols: list[CodeSymbolRef]


class CodeFileText(BaseModel):
    path: str
    language: str
    text: str


class CodeHitOut(_Open):
    node_index: int
    layer: int
    kind: str
    path: str
    start_line: int
    end_line: int
    symbol: str
    score: float
    text: str


class CodeSymbol(BaseModel):
    hit: CodeHitOut
    children: list[CodeHitOut]
    parents: list[CodeHitOut]
    callers: list[CodeHitOut]
    callees: list[CodeHitOut]


class CodeGraphNode(BaseModel):
    id: int | str
    label: str
    kind: str
    path: str
    start_line: int
    layer: int


class CodeGraphEdge(BaseModel):
    src: int | str
    dst: int | str
    kind: str


class CodeGraph(BaseModel):
    nodes: list[CodeGraphNode]
    edges: list[CodeGraphEdge]


class CodeAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    k: int = Field(6, ge=1, le=30)


EventCatalog.model_rebuild()
