"""Stand-in for :class:`raptor.code.CodeIndex` until Phase 3 lands.

It implements the same interface (``build/save/load/update/search/ask/stream_ask/files/
symbol/graph`` plus ``.tree`` and ``.repo_path``) with Python's ``ast`` module instead of
tree-sitter, and extractive summaries instead of LLM ones. It is only used when
``raptor.code`` cannot be imported, so the Code Explorer can be developed and tested.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..models.base import BaseEmbeddingModel, BaseQAModel
from ..storage import load_tree, save_tree
from ..types import Node, Tree

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".cfg", ".yaml", ".yml", ".json"}
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass
class CodeHit:
    node_index: int
    layer: int
    kind: str
    path: str
    start_line: int
    end_line: int
    symbol: str
    score: float
    text: str


@dataclass
class CodeAnswer:
    answer: str
    citations: list[CodeHit] = field(default_factory=list)


@dataclass
class CodeEdge:
    src: int
    dst: int
    kind: str  # "calls" | "imports" | "inherits"


def _split_ident(text: str) -> set[str]:
    out = set()
    for tok in _IDENT.findall(text):
        for part in re.split(r"_|(?<=[a-z])(?=[A-Z])", tok):
            if len(part) > 2:
                out.add(part.lower())
    return out


class StubCodeIndex:
    STUB = True

    def __init__(
        self,
        repo_path: Path,
        tree: Tree,
        edges: list[CodeEdge],
        *,
        embedding_model: BaseEmbeddingModel | None = None,
        qa_model: BaseQAModel | None = None,
    ) -> None:
        self.repo_path = Path(repo_path)
        self.tree = tree
        self.edges = edges
        self._embedding = embedding_model
        self._qa = qa_model

    # ----------------------------------------------------------------- build
    @classmethod
    def build(
        cls,
        repo_path: str | Path,
        config: Any = None,
        *,
        embedding_model: BaseEmbeddingModel | None = None,
        summarization_model: Any = None,
        qa_model: BaseQAModel | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> StubCodeIndex:
        from .fakes import FakeEmbeddingModel, FakeQAModel

        emb = embedding_model or FakeEmbeddingModel()
        root = Path(repo_path)
        progress = progress or (lambda _ev: None)
        files = sorted(
            p
            for p in root.rglob("*")
            if p.is_file()
            and p.suffix in TEXT_SUFFIXES
            and not (set(p.relative_to(root).parts) & SKIP_DIRS)
        )
        progress({"stage": "parsing", "total": len(files)})
        nodes: list[Node] = []
        by_name: dict[str, list[int]] = {}
        module_of: dict[str, int] = {}
        file_nodes: dict[str, int] = {}
        calls: list[tuple[int, str]] = []
        file_imports: list[list[str]] = []
        bases: list[tuple[int, str]] = []

        def add(
            text: str, layer: int, meta: dict[str, Any], children: set[int] | None = None
        ) -> int:
            n = Node(text, len(nodes), children or set(), {}, layer, meta)
            nodes.append(n)
            return n.index

        leaf_ids: list[int] = []
        file_specs: list[tuple[str, str, list[int]]] = []
        for f in files:
            rel = f.relative_to(root).as_posix()
            source = f.read_text(encoding="utf-8", errors="replace")
            lines = source.splitlines()
            symbols: list[int] = []
            mods: list[str] = []
            if f.suffix == ".py":
                try:
                    mod = ast.parse(source)
                except SyntaxError:
                    mod = ast.Module(body=[], type_ignores=[])
                for item in ast.walk(mod):
                    if isinstance(item, (ast.Import, ast.ImportFrom)):
                        names = (
                            [a.name for a in item.names]
                            if isinstance(item, ast.Import)
                            else [item.module or ""]
                        )
                        mods.extend(n for n in names if n)
                for top in mod.body:
                    for sym, parent in _symbols(top):
                        end = getattr(sym, "end_lineno", sym.lineno) or sym.lineno
                        qual = f"{parent}.{sym.name}" if parent else sym.name
                        kind = (
                            "class"
                            if isinstance(sym, ast.ClassDef)
                            else "method"
                            if parent
                            else "function"
                        )
                        body = "\n".join(lines[sym.lineno - 1 : min(end, sym.lineno + 59)])
                        doc = ast.get_docstring(sym) or ""
                        header = f"{rel} › {qual}"
                        nid = add(
                            f"# {header}\n{body}",
                            0,
                            {
                                "kind": kind,
                                "path": rel,
                                "start_line": sym.lineno,
                                "end_line": end,
                                "symbol": qual,
                                "docstring": doc,
                            },
                        )
                        symbols.append(nid)
                        by_name.setdefault(sym.name, []).append(nid)
                        for sub in ast.walk(sym):
                            if isinstance(sub, ast.Call):
                                fn = sub.func
                                name = (
                                    fn.id
                                    if isinstance(fn, ast.Name)
                                    else fn.attr
                                    if isinstance(fn, ast.Attribute)
                                    else None
                                )
                                if name and name != sym.name:
                                    calls.append((nid, name))
                        if isinstance(sym, ast.ClassDef):
                            for b in sym.bases:
                                if isinstance(b, ast.Name):
                                    bases.append((nid, b.id))
                doc = ast.get_docstring(mod) or ""
            else:
                doc = " ".join(lines[:3])
            file_specs.append((rel, doc, symbols))
            file_imports.append(mods)
            leaf_ids.extend(symbols)

        # layer 1: files (a file with no symbols becomes its own leaf-level summary)
        imports: list[tuple[int, str]] = []
        for pos, (rel, doc, symbols) in enumerate(file_specs):
            names = [nodes[i].metadata["symbol"] for i in symbols]
            text = f"File {rel}. {doc.strip()} " + (
                f"Defines {', '.join(names[:12])}." if names else ""
            )
            nid = add(
                text.strip(),
                1 if symbols else 0,
                {
                    "kind": "file",
                    "path": rel,
                    "start_line": 1,
                    "end_line": len(
                        (root / rel).read_text(encoding="utf-8", errors="replace").splitlines()
                    ),
                    "symbol": rel,
                },
                set(symbols),
            )
            file_nodes[rel] = nid
            if rel.endswith(".py"):
                module_of[rel[:-3].replace("/", ".").removesuffix(".__init__")] = nid
            imports.extend((nid, m) for m in file_imports[pos])
        # layer 2: directories
        dirs: dict[str, set[int]] = {}
        for rel, nid in file_nodes.items():
            d = rel.rsplit("/", 1)[0] if "/" in rel else "."
            dirs.setdefault(d, set()).add(nid)
        dir_ids = []
        for d, members in sorted(dirs.items()):
            names = sorted(nodes[i].metadata["path"] for i in members)
            dir_ids.append(
                add(
                    f"Directory {d} with {len(names)} files: {', '.join(names)}.",
                    2,
                    {"kind": "directory", "path": d, "start_line": 0, "end_line": 0, "symbol": d},
                    members,
                )
            )
        add(
            f"Repository {root.name}: {len(file_nodes)} files, {len(leaf_ids)} symbols.",
            3,
            {"kind": "repo", "path": ".", "start_line": 0, "end_line": 0, "symbol": root.name},
            set(dir_ids),
        )

        edges: list[CodeEdge] = []
        for src, name in calls:
            for callee in by_name.get(name, [])[:3]:
                edges.append(CodeEdge(src, callee, "calls"))
        for src, name in bases:
            for base in by_name.get(name, [])[:1]:
                edges.append(CodeEdge(src, base, "inherits"))
        for src, mod_name in imports:
            dst: int | None = module_of.get(mod_name)
            if dst is None:
                # relative-ish: match on the last component
                tail = mod_name.rsplit(".", 1)[-1]
                dst = next((v for k, v in module_of.items() if k.endswith(tail) and v >= 0), None)
            if dst is not None and dst >= 0 and src >= 0 and src != dst:
                edges.append(CodeEdge(src, dst, "imports"))
        edges = list({(e.src, e.dst, e.kind): e for e in edges}.values())

        progress({"stage": "embedding", "total": len(nodes)})
        vecs = emb.embed_documents([n.text for n in nodes])
        for n, v in zip(nodes, vecs, strict=True):
            n.embeddings = {"EMB": np.asarray(v, dtype=np.float32)}
        layers: dict[int, list[Node]] = {}
        for n in nodes:
            layers.setdefault(n.layer, []).append(n)
        tree = Tree.from_layers(
            dict(sorted(layers.items())),
            {"embedding_key": "EMB", "embedding_model": emb.model_id, "kind": "code", "stub": True},
        )
        progress({"stage": "done", "num_nodes": len(nodes)})
        return cls(root, tree, edges, embedding_model=emb, qa_model=qa_model or FakeQAModel())

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        save_tree(self.tree, path)
        meta = {"repo_path": str(self.repo_path), "edges": [asdict(e) for e in self.edges]}
        (path / "code.json").write_text(json.dumps(meta), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> StubCodeIndex:
        from .fakes import FakeEmbeddingModel, FakeQAModel

        path = Path(path)
        meta = json.loads((path / "code.json").read_text(encoding="utf-8"))
        return cls(
            Path(meta["repo_path"]),
            load_tree(path),
            [CodeEdge(**e) for e in meta["edges"]],
            embedding_model=FakeEmbeddingModel(),
            qa_model=FakeQAModel(),
        )

    def update(self, progress: Callable[[dict[str, Any]], None] | None = None) -> StubCodeIndex:
        fresh = self.build(
            self.repo_path, embedding_model=self._embedding, qa_model=self._qa, progress=progress
        )
        self.tree, self.edges = fresh.tree, fresh.edges
        return self

    # ----------------------------------------------------------------- query
    def _hit(self, idx: int, score: float) -> CodeHit:
        n = self.tree.all_nodes[idx]
        m = n.metadata
        return CodeHit(
            node_index=idx,
            layer=n.layer,
            kind=m.get("kind", "symbol"),
            path=m.get("path", ""),
            start_line=int(m.get("start_line", 0)),
            end_line=int(m.get("end_line", 0)),
            symbol=m.get("symbol", ""),
            score=round(score, 4),
            text=n.text,
        )

    def search(self, q: str, k: int = 8) -> list[CodeHit]:
        assert self._embedding is not None
        indices, matrix = self.tree.normalized_matrix()
        qv = np.asarray(self._embedding.embed_query(q), dtype=np.float32)
        qv /= float(np.linalg.norm(qv)) or 1.0
        dense = matrix @ qv
        q_terms = _split_ident(q)
        lexical = np.asarray(
            [
                len(q_terms & _split_ident(self.tree.all_nodes[i].metadata.get("symbol", "")))
                for i in indices
            ],
            dtype=np.float32,
        )
        score = dense + 0.25 * lexical
        order = np.argsort(-score, kind="stable")[:k]
        return [self._hit(indices[int(i)], float(score[int(i)])) for i in order]

    def stream_ask(self, q: str, k: int = 6) -> Iterator[str | CodeAnswer]:
        assert self._qa is not None
        hits = self.search(q, k)
        context = "\n\n".join(
            f"[{h.path}:{h.start_line}] {h.symbol}\n{h.text[:1200]}" for h in hits
        )
        parts: list[str] = []
        cites = [h for h in hits if h.layer == 0][:4] or hits[:3]
        if self._qa.model_id.startswith("fake:"):
            pieces = self._extractive_answer(cites)
        else:
            pieces = self._qa.stream_answer(context, q)
        for piece in pieces:
            parts.append(piece)
            yield piece
        tail = " See " + ", ".join(f"{h.path}:{h.start_line}" for h in cites) + "."
        parts.append(tail)
        yield tail
        yield CodeAnswer("".join(parts), cites)

    def _extractive_answer(self, cites: list[CodeHit]) -> Iterator[str]:
        """Offline answer: describe the best-matching symbols from their docstrings."""
        if not cites:
            yield "No matching code was found."
            return
        sentences = []
        for i, h in enumerate(cites[:3]):
            doc = self.tree.all_nodes[h.node_index].metadata.get("docstring") or ""
            what = doc.strip().splitlines()[0].rstrip(".") if doc else f"is a {h.kind}"
            lead = "The most relevant code is" if i == 0 else "Related:"
            sentences.append(f"{lead} `{h.symbol}` in {h.path}: {what[:1].lower() + what[1:]}.")
        for word in " ".join(sentences).split(" "):
            yield word + " "

    def ask(self, q: str, k: int = 6) -> CodeAnswer:
        final: CodeAnswer | None = None
        for piece in self.stream_ask(q, k):
            if isinstance(piece, CodeAnswer):
                final = piece
        assert final is not None
        return final

    def files(self) -> list[str]:
        return sorted(
            n.metadata["path"]
            for n in self.tree.all_nodes.values()
            if n.metadata.get("kind") == "file"
        )

    def symbol(self, nid: int) -> dict[str, Any]:
        node = self.tree.all_nodes[nid]
        parents = self.tree.parents_of(nid)
        return {
            "hit": asdict(self._hit(nid, 1.0)),
            "children": [asdict(self._hit(c, 1.0)) for c in sorted(node.children)],
            "parents": [asdict(self._hit(p, 1.0)) for p in parents],
            "callers": [
                asdict(self._hit(e.src, 1.0))
                for e in self.edges
                if e.dst == nid and e.kind == "calls"
            ],
            "callees": [
                asdict(self._hit(e.dst, 1.0))
                for e in self.edges
                if e.src == nid and e.kind == "calls"
            ],
        }

    def graph(self, kinds: list[str] | tuple[str, ...] | None = None) -> list[CodeEdge]:
        want = set(kinds or ("calls", "imports", "inherits"))
        return [e for e in self.edges if e.kind in want]


def _symbols(node: ast.stmt, parent: str = "") -> Iterator[tuple[Any, str]]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        yield node, parent
    elif isinstance(node, ast.ClassDef):
        yield node, parent
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield child, node.name
