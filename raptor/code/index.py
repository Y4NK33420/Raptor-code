"""The code index: build, persist, update incrementally, search and answer.

On disk an index is a directory::

    <dir>/
      tree/           # the raptor Tree (tree.json + embeddings.npz, see raptor.storage)
      code.json       # repo path, config, git head, per-file hashes and parsed symbols
      graph.json      # import / call / inheritance edges between tree nodes
"""

from __future__ import annotations

import json
import logging
import os
import posixpath
import tempfile
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tiktoken

from ..chunking import default_tokenizer
from ..models.base import BaseEmbeddingModel, BaseQAModel, BaseSummarizationModel
from ..storage import load_tree, save_tree
from ..types import Tree
from .build import CodeTreeBuilder, Reuse
from .config import CodeConfig
from .graph import CodeGraphBuilder, SymKey
from .ingest import discover_files, git_changed_files, git_head, read_source, sha256_bytes
from .languages import extractor_for, make_extractors
from .languages.base import ParsedFile
from .models import LazyModels
from .prompts import CODE_QA_INSTRUCTIONS
from .search import CodeSearcher, node_kind, to_hit
from .types import CodeAnswer, CodeEdge, CodeHit

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
TREE_DIR = "tree"
CODE_FILE = "code.json"
GRAPH_FILE = "graph.json"
README_NAMES = ("README.md", "README.rst", "README.txt", "README")

ProgressCallback = Callable[[dict[str, Any]], None]


def _noop(_: dict[str, Any]) -> None:
    pass


def _write_json(path: Path, data: Any) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class CodeIndex:
    """A RAPTOR index over a source repository.

    Build one with :meth:`build`, persist it with :meth:`save` and reopen it with
    :meth:`load`. Node indices are stable for a given index state but may change after
    :meth:`update`.
    """

    def __init__(
        self,
        repo_path: str | Path,
        config: CodeConfig,
        tree: Tree,
        edges: list[CodeEdge],
        files: dict[str, dict[str, Any]],
        *,
        models: LazyModels,
        git_commit: str | None = None,
        dirty: list[str] | None = None,
        tokenizer: tiktoken.Encoding | None = None,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.config = config
        self.tree = tree
        self.edges = edges
        self._files = files
        self.models = models
        self.git_commit = git_commit
        self.dirty = list(dirty or [])
        self.tokenizer = tokenizer or default_tokenizer()
        self.last_build_stats: dict[str, int] = {}
        self._searcher: CodeSearcher | None = None

    # ================================================================ building
    @classmethod
    def build(
        cls,
        repo_path: str | Path,
        config: CodeConfig | None = None,
        *,
        embedding_model: BaseEmbeddingModel | None = None,
        summarization_model: BaseSummarizationModel | None = None,
        qa_model: BaseQAModel | None = None,
        progress: ProgressCallback | None = None,
    ) -> CodeIndex:
        """Parse, summarize and embed the repository at ``repo_path``."""
        config = config or CodeConfig()
        root = Path(repo_path).resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"{root} is not a directory")
        models = LazyModels(
            config.models,
            embedding=embedding_model,
            summarization=summarization_model,
            qa=qa_model,
        )
        index = cls(root, config, Tree.from_layers({0: []}), [], {}, models=models)
        index._rebuild(progress or _noop, previous=None)
        return index

    def _extractors(self):
        return make_extractors(self.config.ingest.languages)

    def _scan(
        self, progress: ProgressCallback, reuse_files: dict[str, dict[str, Any]] | None
    ) -> tuple[dict[str, ParsedFile], dict[str, str], dict[str, dict[str, Any]], list[str]]:
        """Read and parse the repo. Files whose hash is unchanged reuse their stored parse."""
        extractors = self._extractors()
        paths = discover_files(self.repo_path, self.config.ingest, extractors.keys())
        parsed: dict[str, ParsedFile] = {}
        sources: dict[str, str] = {}
        records: dict[str, dict[str, Any]] = {}
        changed: list[str] = []
        total = len(paths)
        progress({"stage": "parsing", "done": 0, "total": total})
        for i, rel in enumerate(paths):
            data = read_source(self.repo_path, rel, self.config.ingest)
            if data is None:
                continue
            sha = sha256_bytes(data)
            prev = (reuse_files or {}).get(rel)
            text = data.decode("utf-8", errors="replace")
            if prev is not None and prev["sha"] == sha:
                pf = ParsedFile.from_dict(prev["parsed"])
            else:
                ex = extractor_for(rel, extractors)
                if ex is None:
                    continue
                try:
                    pf = ex.parse(data, rel)
                except Exception as exc:  # pragma: no cover - parser bugs
                    logger.warning("failed to parse %s: %s", rel, exc)
                    continue
                changed.append(rel)
            parsed[rel] = pf
            sources[rel] = text
            records[rel] = {"sha": sha, "language": pf.language, "parsed": pf.to_dict()}
            if (i + 1) % 50 == 0 or i + 1 == total:
                progress({"stage": "parsing", "done": i + 1, "total": total})
        return parsed, sources, records, changed

    def _readme(self) -> str | None:
        for name in README_NAMES:
            p = self.repo_path / name
            if p.is_file():
                try:
                    return p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    return None
        return None

    def _rebuild(
        self,
        progress: ProgressCallback,
        *,
        previous: CodeIndex | None,
    ) -> tuple[list[str], dict[str, int]]:
        prev_files = previous._files if previous is not None else None
        parsed, sources, records, changed = self._scan(progress, prev_files)
        if not parsed:
            raise ValueError(f"no indexable source files found under {self.repo_path}")
        reuse = (
            Reuse.from_tree(previous.tree, self.models.embedding.model_id)
            if previous is not None
            else Reuse()
        )
        builder = CodeTreeBuilder(
            self.config,
            self.models.embedding,
            self.models.summarization,
            self.tokenizer,
            reuse=reuse,
            progress=progress,
        )
        tree = builder.build(self.repo_path.name, parsed, sources, self._readme())
        self.tree = tree
        self._files = records
        self.edges = self._compute_edges(parsed)
        self.git_commit = git_head(self.repo_path)
        self.dirty = (
            sorted(git_changed_files(self.repo_path, self.git_commit) or set())
            if (self.git_commit)
            else []
        )
        self._searcher = None
        stats = {
            "reembedded": builder.stats.reembedded,
            "resummarized": builder.stats.resummarized,
            "reused_summaries": builder.stats.reused_summaries,
            "reused_embeddings": builder.stats.reused_embeddings,
        }
        self.last_build_stats = stats
        return changed, stats

    def _symbol_nodes(self) -> dict[SymKey, int]:
        """Map ``(path, qualname)`` (and ``(path, None)`` for files) to node indices."""
        out: dict[SymKey, int] = {}
        for n in sorted(self.tree.all_nodes.values(), key=lambda n: n.index):
            m = n.metadata
            kind = m.get("kind")
            if kind == "file":
                out[(m["path"], None)] = n.index
            elif kind == "class":
                out[(m["path"], m["symbol"])] = n.index
        for n in sorted(self.tree.leaf_nodes.values(), key=lambda n: n.index):
            m = n.metadata
            for q in m.get("symbols", []):
                out.setdefault((m["path"], q), n.index)
        return out

    def _compute_edges(self, parsed: dict[str, ParsedFile]) -> list[CodeEdge]:
        sym_edges = CodeGraphBuilder(parsed, self._extractors()).edges()
        nodes = self._symbol_nodes()
        out: set[CodeEdge] = set()
        for e in sym_edges:
            src, dst = nodes.get(e.src), nodes.get(e.dst)
            if src is not None and dst is not None and src != dst:
                out.add(CodeEdge(src, dst, e.kind))
        return sorted(out, key=lambda e: (e.kind, e.src, e.dst))

    # ================================================================ updating
    def update(self, progress: ProgressCallback | None = None) -> dict[str, Any]:
        """Re-index what changed since the last build/update.

        Changed files are found with ``git diff`` against the indexed commit when the repo
        is a git checkout (confirmed by content hash), otherwise by hashing every file.
        Only changed symbols are re-embedded and only their ancestors are re-summarized.
        """
        progress = progress or _noop
        candidates: set[str] | None = None
        if self.git_commit and git_head(self.repo_path) is not None:
            diff = git_changed_files(self.repo_path, self.git_commit)
            if diff is not None:
                candidates = diff | set(self.dirty)
        old_files = dict(self._files)
        current = set(discover_files(self.repo_path, self.config.ingest, self._extractors().keys()))
        added = current - old_files.keys()
        deleted = old_files.keys() - current
        to_check = (candidates & current) | added if candidates is not None else current
        changed: list[str] = []
        for rel in sorted(to_check):
            data = read_source(self.repo_path, rel, self.config.ingest)
            sha = sha256_bytes(data) if data is not None else None
            if sha is None or rel not in old_files or old_files[rel]["sha"] != sha:
                changed.append(rel)
        changed_all = sorted(set(changed) | deleted)
        if not changed_all:
            return {"changed_files": [], "reembedded": 0, "resummarized": 0}
        previous = CodeIndex(
            self.repo_path,
            self.config,
            self.tree,
            self.edges,
            old_files,
            models=self.models,
            git_commit=self.git_commit,
        )
        # Files with an unchanged hash reuse their stored parse; only changed ones are parsed.
        _, stats = self._rebuild(progress, previous=previous)
        return {
            "changed_files": changed_all,
            "reembedded": stats["reembedded"],
            "resummarized": stats["resummarized"],
        }

    # ============================================================ persistence
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_tree(self.tree, path / TREE_DIR)
        _write_json(
            path / CODE_FILE,
            {
                "format_version": FORMAT_VERSION,
                "repo_path": str(self.repo_path),
                "config": self.config.model_dump(mode="json"),
                "git_commit": self.git_commit,
                "dirty": self.dirty,
                "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "files": self._files,
            },
        )
        _write_json(path / GRAPH_FILE, {"edges": [e.to_dict() for e in self.edges]})
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repo_path: str | Path | None = None,
        config: CodeConfig | None = None,
        embedding_model: BaseEmbeddingModel | None = None,
        summarization_model: BaseSummarizationModel | None = None,
        qa_model: BaseQAModel | None = None,
    ) -> CodeIndex:
        path = Path(path)
        code_file = path / CODE_FILE
        if not code_file.exists():
            raise FileNotFoundError(f"no code index at {path} (missing {CODE_FILE})")
        doc = json.loads(code_file.read_text(encoding="utf-8"))
        if doc.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported code index format {doc.get('format_version')!r}")
        cfg = config or CodeConfig.model_validate(doc["config"])
        tree = load_tree(path / TREE_DIR)
        graph = json.loads((path / GRAPH_FILE).read_text(encoding="utf-8"))
        edges = [CodeEdge(int(e["src"]), int(e["dst"]), str(e["kind"])) for e in graph["edges"]]
        models = LazyModels(
            cfg.models,
            embedding=embedding_model,
            summarization=summarization_model,
            qa=qa_model,
        )
        return cls(
            repo_path or doc["repo_path"],
            cfg,
            tree,
            edges,
            doc["files"],
            models=models,
            git_commit=doc.get("git_commit"),
            dirty=doc.get("dirty", []),
        )

    # ================================================================ queries
    @property
    def searcher(self) -> CodeSearcher:
        if self._searcher is None:
            emb = self.models.embedding if self.config.retrieval.dense else None
            self._searcher = CodeSearcher(self.tree, self.edges, emb, self.config.retrieval)
        return self._searcher

    def search(self, query: str, k: int = 10, **options: Any) -> list[CodeHit]:
        """Hybrid search. ``options`` override retrieval settings for this call
        (``dense``, ``bm25``, ``graph_expansion``, ``descend``, ``routing``, ``kinds``)."""
        return self.searcher.search(query, k, **options)

    def _context(self, hits: list[CodeHit]) -> tuple[str, list[CodeHit]]:
        budget = self.config.retrieval.max_context_tokens
        parts: list[str] = []
        used: list[CodeHit] = []
        total = 0
        for h in hits:
            block = f"[{h.location}] ({h.kind} {h.symbol})\n{h.text.strip()}\n"
            n = len(self.tokenizer.encode(block, disallowed_special=()))
            if total + n > budget:
                if used:
                    continue
                block = self.tokenizer.decode(
                    self.tokenizer.encode(block, disallowed_special=())[:budget]
                )
                n = budget
            parts.append(block)
            used.append(h)
            total += n
        return CODE_QA_INSTRUCTIONS + "\n\n" + "\n".join(parts), used

    def _ask_hits(self, question: str, k: int) -> list[CodeHit]:
        hits = self.search(question, k)
        # Graph context: add the callers / callees of the top symbol hits if budget allows.
        seen = {h.node_index for h in hits}
        extra: list[CodeHit] = []
        if self.config.retrieval.graph_expansion:
            for h in hits[:3]:
                if h.kind not in ("symbol", "class"):
                    continue
                nb = self.searcher.neighbors(h.node_index)
                for idx in (nb["callees"] + nb["callers"])[:2]:
                    if idx not in seen:
                        seen.add(idx)
                        extra.append(to_hit(self.tree.all_nodes[idx], 0.0))
        return hits + extra

    def ask(self, question: str, k: int = 10) -> CodeAnswer:
        context, used = self._context(self._ask_hits(question, k))
        answer = self.models.qa.answer_question(context, question)
        return CodeAnswer(answer=answer, citations=used)

    def stream_ask(self, question: str, k: int = 10) -> Iterator[str | CodeAnswer]:
        """Yield the answer as text pieces, then a final :class:`CodeAnswer`."""
        context, used = self._context(self._ask_hits(question, k))
        pieces: list[str] = []
        for piece in self.models.qa.stream_answer(context, question):
            pieces.append(piece)
            yield piece
        yield CodeAnswer(answer="".join(pieces), citations=used)

    # =============================================================== browsing
    def files(self) -> list[str]:
        return sorted(self._files)

    def node_hit(self, node_index: int, score: float = 0.0) -> CodeHit:
        return to_hit(self.tree.all_nodes[node_index], score)

    def symbol(self, node_index: int) -> dict[str, Any]:
        """A node with its children, parents, callers and callees (as :class:`CodeHit`)."""
        if node_index not in self.tree.all_nodes:
            raise KeyError(f"no node {node_index}")
        node = self.tree.all_nodes[node_index]
        callers, callees = self._call_neighbors(node_index)
        return {
            "hit": self.node_hit(node_index),
            "children": [self.node_hit(c) for c in sorted(node.children)],
            "parents": [self.node_hit(p) for p in self.searcher.parents.get(node_index, [])],
            "callers": [self.node_hit(i) for i in callers],
            "callees": [self.node_hit(i) for i in callees],
        }

    def _call_neighbors(self, node_index: int) -> tuple[list[int], list[int]]:
        node = self.tree.all_nodes[node_index]
        kind = node_kind(node)
        s = self.searcher
        if kind == "file":
            imports_out = [e.dst for e in self.edges if e.kind == "imports" and e.src == node_index]
            imports_in = [e.src for e in self.edges if e.kind == "imports" and e.dst == node_index]
            return imports_in, imports_out
        members = {node_index}
        if kind == "class":
            members |= set(node.children)
        callers = [c for m in members for c in s.callers.get(m, []) if c not in members]
        callees = [c for m in members for c in s.callees.get(m, []) if c not in members]
        return list(dict.fromkeys(callers)), list(dict.fromkeys(callees))

    def graph(self, kinds: Iterable[str] = ("calls", "imports", "inherits")) -> list[CodeEdge]:
        wanted = set(kinds)
        return [e for e in self.edges if e.kind in wanted]

    def find_symbol(self, name: str) -> list[CodeHit]:
        """Nodes whose symbol, qualified name or path matches ``name`` exactly (or by suffix)."""
        name = name.strip()
        out = []
        for n in sorted(self.tree.all_nodes.values(), key=lambda n: (n.layer, n.index)):
            m = n.metadata
            names = [str(m.get("symbol", "")), *m.get("symbols", [])]
            path = str(m.get("path", ""))
            if (
                name in names
                or any(x.endswith("." + name) for x in names)
                or (
                    node_kind(n) in ("file", "dir")
                    and (path == name or path.rstrip("/") == name.rstrip("/"))
                )
            ):
                out.append(self.node_hit(n.index))
        return out

    def module_node(self, path: str) -> int | None:
        """Node index of the file or directory ``path`` (or the repo for ``.``/empty)."""
        path = path.strip().strip("/").replace("\\", "/")
        for n in self.tree.all_nodes.values():
            m = n.metadata
            if path in ("", ".") and m.get("kind") == "repo":
                return n.index
            if m.get("kind") in ("file", "dir") and m.get("path") == path:
                return n.index
        # Accept a dotted module name, e.g. "raptor.code.index".
        dotted = path.replace(".", "/")
        for cand in (f"{dotted}.py", f"{dotted}/__init__.py", dotted):
            for n in self.tree.all_nodes.values():
                m = n.metadata
                if m.get("kind") in ("file", "dir") and (
                    m.get("path") == cand or str(m.get("path", "")).endswith("/" + cand)
                ):
                    return n.index
        return None

    def outline(self, max_depth: int = 4) -> list[tuple[int, CodeHit]]:
        """``(depth, hit)`` rows of the structural hierarchy, repo first (for printing)."""
        repo = next(n for n in self.tree.root_nodes.values() if n.metadata.get("kind") == "repo")
        rows: list[tuple[int, CodeHit]] = []

        def visit(idx: int, depth: int) -> None:
            node = self.tree.all_nodes[idx]
            rows.append((depth, self.node_hit(idx)))
            if depth >= max_depth:
                return
            kids = sorted(
                node.children,
                key=lambda c: (
                    self.tree.all_nodes[c].metadata.get("kind") != "dir",
                    str(self.tree.all_nodes[c].metadata.get("path", "")),
                    self.tree.all_nodes[c].metadata.get("start_line", 0),
                ),
            )
            for c in kids:
                visit(c, depth + 1)

        visit(repo.index, 0)
        for n in sorted(self.tree.root_nodes.values(), key=lambda n: n.index):
            if n.metadata.get("kind") == "concept":
                rows.append((0, self.node_hit(n.index)))
        return rows

    def stats(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for n in self.tree.all_nodes.values():
            k = node_kind(n)
            counts[k] = counts.get(k, 0) + 1
        edges: dict[str, int] = {}
        for e in self.edges:
            edges[e.kind] = edges.get(e.kind, 0) + 1
        return {
            "repo_path": str(self.repo_path),
            "files": len(self._files),
            "nodes": counts,
            "edges": edges,
            "layers": self.tree.num_layers + 1,
            "git_commit": self.git_commit,
        }

    def __repr__(self) -> str:
        return f"CodeIndex({posixpath.basename(str(self.repo_path))!r}, nodes={len(self.tree.all_nodes)})"
