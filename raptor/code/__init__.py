"""RAPTOR for codebases: symbol-level trees, code graph, hybrid search and Q&A.

>>> from raptor.code import CodeIndex
>>> index = CodeIndex.build("path/to/repo")          # doctest: +SKIP
>>> index.save("indexes/repo")                        # doctest: +SKIP
>>> index.ask("How are requests retried?").answer    # doctest: +SKIP
"""

from __future__ import annotations

from .config import CodeConfig
from .index import CodeIndex
from .types import CodeAnswer, CodeEdge, CodeHit

__all__ = ["CodeAnswer", "CodeConfig", "CodeEdge", "CodeHit", "CodeIndex"]
