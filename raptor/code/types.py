"""Public result types of the code index."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

NODE_KINDS = ("symbol", "class", "file", "dir", "repo", "concept")
SUMMARY_KINDS = ("class", "file", "dir", "repo", "concept")


@dataclass
class CodeHit:
    """One retrieved node. ``start_line``/``end_line`` are 0 for dir, repo and concept nodes."""

    node_index: int
    layer: int
    kind: str  # "symbol" | "class" | "file" | "dir" | "repo" | "concept"
    path: str
    start_line: int
    end_line: int
    symbol: str
    score: float
    text: str

    @property
    def location(self) -> str:
        """Citation string: ``path:start-end`` (or ``path/`` / the concept name)."""
        if self.kind in ("symbol", "class", "file") and self.start_line:
            return f"{self.path}:{self.start_line}-{self.end_line}"
        if self.kind == "dir":
            return f"{self.path}/"
        if self.kind == "repo":
            return "(repository overview)"
        return f"(concept: {self.symbol})"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["location"] = self.location
        return d


@dataclass
class CodeAnswer:
    answer: str
    citations: list[CodeHit] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"answer": self.answer, "citations": [c.to_dict() for c in self.citations]}


@dataclass(frozen=True)
class CodeEdge:
    src: int
    dst: int
    kind: str  # "calls" | "imports" | "inherits"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
