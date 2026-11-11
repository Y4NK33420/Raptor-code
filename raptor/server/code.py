"""Code indexes for the Code Explorer, backed by :mod:`raptor.code` when it is installed.

Until Phase 3 merges, :class:`~raptor.server.code_stub.StubCodeIndex` (same interface)
indexes the configured fixture repositories on the fly.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..models import ModelBundle
from .code_stub import StubCodeIndex

logger = logging.getLogger(__name__)
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,120}$")

try:
    from ..code import CodeIndex as _RealCodeIndex  # type: ignore[import-not-found,attr-defined,unused-ignore]

    HAVE_CODE = True
except Exception:  # pragma: no cover - depends on Phase 3
    _RealCodeIndex = None
    HAVE_CODE = False

LANGUAGES = {
    ".py": "python",
    ".md": "markdown",
    ".toml": "toml",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".rs": "rust",
    ".go": "go",
}


def as_dict(obj: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return dict(obj.model_dump())
    return dict(vars(obj))


class CodeService:
    def __init__(
        self,
        code_dir: Path,
        stub_repos: list[Path],
        models: Callable[[], ModelBundle],
        *,
        force_stub: bool = False,
    ) -> None:
        self.code_dir = code_dir
        self.stub_repos = stub_repos
        self.models = models
        self.real = HAVE_CODE and not force_stub
        self._loaded: dict[str, Any] = {}
        self._lock = threading.Lock()

    @property
    def index_cls(self) -> Any:
        return _RealCodeIndex if self.real else StubCodeIndex

    def _sources(self) -> dict[str, tuple[str, Path]]:
        out: dict[str, tuple[str, Path]] = {}
        if self.code_dir.is_dir():
            for d in sorted(self.code_dir.iterdir()):
                if d.is_dir() and _SAFE.match(d.name):
                    out[d.name] = ("saved", d)
        if not self.real:
            for repo in self.stub_repos:
                if repo.is_dir():
                    rid = f"{repo.name}-stub"
                    out.setdefault(rid, ("stub-repo", repo))
        return out

    def get(self, code_id: str) -> Any:
        if not _SAFE.match(code_id):
            raise KeyError(code_id)
        with self._lock:
            if code_id in self._loaded:
                return self._loaded[code_id]
        sources = self._sources()
        if code_id not in sources:
            raise KeyError(code_id)
        kind, path = sources[code_id]
        if kind == "saved":
            idx = self.index_cls.load(path)
        else:
            bundle = self.models()
            idx = StubCodeIndex.build(path, embedding_model=bundle.embedding, qa_model=bundle.qa)
        with self._lock:
            self._loaded[code_id] = idx
        return idx

    def list_all(self) -> list[dict[str, Any]]:
        out = []
        for code_id, (kind, path) in self._sources().items():
            try:
                idx = self.get(code_id)
            except Exception as exc:
                logger.warning("cannot load code index %s: %s", path, exc)
                continue
            tree = idx.tree
            n_sym = sum(1 for n in tree.leaf_nodes.values() if n.metadata.get("kind") != "file")
            out.append(
                {
                    "id": code_id,
                    "name": Path(str(idx.repo_path)).name or code_id,
                    "repo_path": str(idx.repo_path),
                    "n_files": len(idx.files()),
                    "n_symbols": n_sym,
                    "n_nodes": len(tree.all_nodes),
                    "num_layers": tree.num_layers,
                    "stub": bool(getattr(idx, "STUB", False)),
                    "origin": kind,
                }
            )
        return out

    # ---------------------------------------------------------------- queries
    def files(self, code_id: str) -> list[dict[str, Any]]:
        idx = self.get(code_id)
        symbols: dict[str, list[dict[str, Any]]] = {}
        for n in idx.tree.all_nodes.values():
            m = n.metadata
            if m.get("kind") in {"function", "method", "class", "symbol"} and m.get("path"):
                symbols.setdefault(m["path"], []).append(
                    {
                        "id": n.index,
                        "symbol": m.get("symbol", ""),
                        "kind": m.get("kind", "symbol"),
                        "start_line": int(m.get("start_line", 0)),
                        "end_line": int(m.get("end_line", 0)),
                    }
                )
        return [
            {
                "path": p,
                "language": LANGUAGES.get(Path(p).suffix, "text"),
                "symbols": sorted(symbols.get(p, []), key=lambda s: s["start_line"]),
            }
            for p in idx.files()
        ]

    def file_text(self, code_id: str, path: str) -> dict[str, Any]:
        idx = self.get(code_id)
        if path not in set(idx.files()):
            raise KeyError(path)
        root = Path(str(idx.repo_path)).resolve()
        full = (root / path).resolve()
        if root not in full.parents:
            raise KeyError(path)
        text = full.read_text(encoding="utf-8", errors="replace")
        return {"path": path, "language": LANGUAGES.get(full.suffix, "text"), "text": text}

    def symbol(self, code_id: str, nid: int) -> dict[str, Any]:
        out = self.get(code_id).symbol(nid)
        return {k: (as_dict(v) if k == "hit" else [as_dict(x) for x in v]) for k, v in out.items()}

    def graph(self, code_id: str, kinds: list[str] | None) -> dict[str, Any]:
        idx = self.get(code_id)
        edges = [as_dict(e) for e in idx.graph(kinds or ["calls", "imports", "inherits"])]
        ids = {e["src"] for e in edges} | {e["dst"] for e in edges}
        nodes = []
        for i in sorted(ids, key=str):
            n = idx.tree.all_nodes.get(i) if isinstance(i, int) else None
            m = n.metadata if n is not None else {}
            nodes.append(
                {
                    "id": i,
                    "label": m.get("symbol") or str(i),
                    "kind": m.get("kind", "symbol"),
                    "path": m.get("path", ""),
                    "start_line": int(m.get("start_line", 0) or 0),
                    "layer": n.layer if n is not None else 0,
                }
            )
        return {"nodes": nodes, "edges": edges}
