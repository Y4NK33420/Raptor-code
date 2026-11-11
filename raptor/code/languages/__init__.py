"""Language extractors. Python is built in; register more with ``@EXTRACTORS.register``."""

from __future__ import annotations

from .base import EXTRACTORS, ImportRef, LanguageExtractor, ParsedFile, Symbol
from .python import PythonExtractor

__all__ = [
    "EXTRACTORS",
    "ImportRef",
    "LanguageExtractor",
    "ParsedFile",
    "PythonExtractor",
    "Symbol",
    "extractor_for",
    "make_extractors",
]


def make_extractors(languages: list[str]) -> dict[str, LanguageExtractor]:
    """Instantiate the extractors for ``languages``; returns ``{extension: extractor}``."""
    out: dict[str, LanguageExtractor] = {}
    for name in languages:
        ex = EXTRACTORS.create(name)
        for ext in ex.extensions:
            out[ext] = ex
    return out


def extractor_for(path: str, extractors: dict[str, LanguageExtractor]) -> LanguageExtractor | None:
    lower = path.lower()
    for ext, ex in extractors.items():
        if lower.endswith(ext):
            return ex
    return None
