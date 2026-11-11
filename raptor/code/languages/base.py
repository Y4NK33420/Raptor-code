"""Per-language symbol extraction interface.

An extractor turns one source file into a :class:`ParsedFile`: the imports, the symbols
(functions, methods, classes) with their line spans, signatures and docstrings, and the
raw references each symbol makes (calls, base classes). Name resolution across files
happens later in :mod:`raptor.code.graph`, so extractors stay purely syntactic.

Add a language by subclassing :class:`LanguageExtractor` and registering it::

    @EXTRACTORS.register("go")
    class GoExtractor(LanguageExtractor):
        name = "go"
        extensions = (".go",)
        def parse(self, source: bytes, path: str) -> ParsedFile: ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from ...registry import Registry


@dataclass
class ImportRef:
    """One imported name. ``from a.b import c as d`` -> module="a.b", name="c", alias="d"."""

    module: str
    name: str | None = None
    alias: str | None = None
    level: int = 0  # leading dots of a relative import
    line: int = 0
    text: str = ""

    @property
    def bound_name(self) -> str:
        """The local name this import introduces."""
        if self.alias:
            return self.alias
        if self.name:
            return self.name
        return self.module.split(".")[0]


@dataclass
class Symbol:
    """A function, method or class found in a file."""

    qualname: str  # e.g. "Session.request"
    name: str
    kind: str  # "function" | "method" | "class"
    start_line: int  # 1-based, inclusive (includes decorators)
    end_line: int
    signature: str
    docstring: str = ""
    decorators: list[str] = field(default_factory=list)
    parent: str | None = None  # qualname of the enclosing class
    text: str = ""  # source of the symbol (for classes: header + non-method body)
    calls: list[str] = field(default_factory=list)  # raw callee expressions, e.g. "self.send"
    bases: list[str] = field(default_factory=list)  # classes only
    header_end_line: int | None = None  # classes: last line of the header leaf

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Symbol:
        return cls(**data)


@dataclass
class ParsedFile:
    path: str
    language: str
    module_docstring: str = ""
    imports: list[ImportRef] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)
    # 1-based line ranges of top-level code that is not inside any symbol.
    module_code: list[tuple[int, int]] = field(default_factory=list)
    num_lines: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "language": self.language,
            "module_docstring": self.module_docstring,
            "imports": [asdict(i) for i in self.imports],
            "symbols": [s.to_dict() for s in self.symbols],
            "module_code": [list(r) for r in self.module_code],
            "num_lines": self.num_lines,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParsedFile:
        return cls(
            path=data["path"],
            language=data["language"],
            module_docstring=data.get("module_docstring", ""),
            imports=[ImportRef(**i) for i in data.get("imports", [])],
            symbols=[Symbol.from_dict(s) for s in data.get("symbols", [])],
            module_code=[(int(a), int(b)) for a, b in data.get("module_code", [])],
            num_lines=int(data.get("num_lines", 0)),
        )


class LanguageExtractor(ABC):
    name: str = ""
    extensions: tuple[str, ...] = ()

    @abstractmethod
    def parse(self, source: bytes, path: str) -> ParsedFile: ...

    def module_name(self, path: str) -> str | None:
        """Dotted module name for ``path`` (used to resolve imports), or ``None``."""
        return None


EXTRACTORS: Registry[LanguageExtractor] = Registry("language extractor")
