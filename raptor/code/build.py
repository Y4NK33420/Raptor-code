"""Build the code tree: symbol leaves, structural summaries and concept nodes.

Layers (a DAG; a node may have several parents)::

    0          symbol leaves (functions, methods, class headers, module code)
    1          classes
    2          files
    3 .. 3+h   directories (by height: a directory sits above all its subdirectories)
    R          the repository
    R + 1      concepts (clusters of file and class nodes, cutting across directories)

Every summary node records ``input_hash`` (a hash of its prompt). When an index is
rebuilt or updated, a node whose prompt is unchanged reuses its previous summary and
embedding, so only the ancestors of changed code are summarized again.
"""

from __future__ import annotations

import logging
import posixpath
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import tiktoken

from ..clustering import make_clusterer
from ..models.base import BaseEmbeddingModel, BaseSummarizationModel
from ..types import Node, Tree
from . import prompts
from .chunks import Leaf, make_leaves, short_hash
from .config import CodeConfig
from .languages.base import ParsedFile

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, Any]], None]

#: Files with less source than this get an extractive summary (no LLM call).
TINY_FILE_TOKENS = 60
README_TOKENS = 800


@dataclass
class Reuse:
    """What a rebuild may take from the previous index."""

    embeddings: dict[str, np.ndarray] = field(default_factory=dict)  # text hash -> vector
    summaries: dict[str, str] = field(default_factory=dict)  # prompt hash -> summary
    concepts: list[list[str]] | None = None  # previous concept memberships (node keys)
    concept_pool: set[str] = field(default_factory=set)  # keys that were clustered last time

    @classmethod
    def from_tree(cls, tree: Tree, embedding_model_id: str | None = None) -> Reuse:
        r = cls()
        key = tree.embedding_key
        same_model = embedding_model_id is None or tree.metadata.get("embedding_model") in (
            None,
            embedding_model_id,
        )
        concepts: list[list[str]] = []
        for n in tree.all_nodes.values():
            if same_model and key in n.embeddings:
                r.embeddings[short_hash(n.text)] = np.asarray(n.embeddings[key], np.float32)
            ih = n.metadata.get("input_hash")
            if ih and "summary" in n.metadata:
                r.summaries[ih] = n.metadata["summary"]
            if n.metadata.get("kind") == "concept":
                concepts.append(
                    [tree.all_nodes[c].metadata.get("key", "") for c in sorted(n.children)]
                )
        r.concepts = concepts or None
        r.concept_pool = set(tree.metadata.get("concept_pool", []))
        return r


@dataclass
class BuildStats:
    reembedded: int = 0
    resummarized: int = 0
    reused_summaries: int = 0
    reused_embeddings: int = 0


def _first_line(text: str, limit: int = 160) -> str:
    line = text.strip().split("\n", 1)[0] if text.strip() else ""
    return line if len(line) <= limit else line[: limit - 3] + "..."


class CodeTreeBuilder:
    def __init__(
        self,
        config: CodeConfig,
        embedding_model: BaseEmbeddingModel,
        summarization_model: BaseSummarizationModel,
        tokenizer: tiktoken.Encoding,
        *,
        reuse: Reuse | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.cfg = config
        self.emb = embedding_model
        self.summ = summarization_model
        self.tok = tokenizer
        self.reuse = reuse or Reuse()
        self.progress = progress or (lambda _ev: None)
        self.stats = BuildStats()
        self._nodes: list[Node] = []
        self.concept_pool: list[str] = []

    # ------------------------------------------------------------------ utils
    def _ntok(self, text: str) -> int:
        return len(self.tok.encode(text, disallowed_special=()))

    def _truncate(self, text: str, max_tokens: int) -> str:
        ids = self.tok.encode(text, disallowed_special=())
        if len(ids) <= max_tokens:
            return text
        return self.tok.decode(ids[:max_tokens]) + "\n... (truncated)"

    def _new_node(self, text: str, layer: int, children: set[int], meta: dict[str, Any]) -> Node:
        node = Node(text, len(self._nodes), set(children), {}, layer, meta)
        self._nodes.append(node)
        return node

    def _embed(self, nodes: Sequence[Node], layer: int) -> None:
        key = self.cfg.embedding_key
        todo: list[Node] = []
        for n in nodes:
            hit = self.reuse.embeddings.get(short_hash(n.text))
            if hit is not None:
                n.embeddings[key] = hit
                self.stats.reused_embeddings += 1
            else:
                todo.append(n)
        self.progress({"stage": "embedding", "layer": layer, "total": len(todo)})
        if todo:
            vectors = self.emb.embed_documents([n.text for n in todo])
            for n, v in zip(todo, vectors, strict=True):
                n.embeddings[key] = np.asarray(v, dtype=np.float32)
            self.stats.reembedded += len(todo)

    def _summarize_many(self, prompts_: list[str], layer: int, kind: str) -> list[str]:
        """Summaries for ``prompts_``, reusing previous ones for unchanged prompts."""
        out: list[str | None] = []
        todo: list[int] = []
        for i, p in enumerate(prompts_):
            prev = self.reuse.summaries.get(short_hash(p))
            out.append(prev)
            if prev is None:
                todo.append(i)
            else:
                self.stats.reused_summaries += 1
        total = len(todo)
        self.progress(
            {"stage": "summarizing", "layer": layer, "kind": kind, "done": 0, "total": total}
        )
        if todo:
            length = int(self.cfg.summary.length * 1.6)

            def run(i: int) -> str:
                return self.summ.summarize(prompts_[i], max_tokens=length).strip()

            done = 0
            with ThreadPoolExecutor(max_workers=self.cfg.summary.max_workers) as pool:
                futures = {pool.submit(run, i): i for i in todo}
                for fut, i in futures.items():
                    out[i] = fut.result()
                    done += 1
                    self.progress(
                        {
                            "stage": "summarizing",
                            "layer": layer,
                            "kind": kind,
                            "done": done,
                            "total": total,
                        }
                    )
            self.stats.resummarized += len(todo)
        return [o or "" for o in out]

    def _fmt_prompt(self, template: str, **kw: Any) -> str:
        return template.format(max_words=self.cfg.summary.length, **kw)

    def _fit_children(self, items: list[tuple[str, str]], budget: int) -> str:
        """Format ``(label, summary)`` pairs, trimming each to share ``budget`` tokens."""
        if not items:
            return "(empty)"
        per = max(40, budget // len(items))
        return "\n".join(f"- {label}: {self._truncate(s, per)}" for label, s in items)

    # ------------------------------------------------------------------ build
    def build(
        self,
        repo_name: str,
        parsed: dict[str, ParsedFile],
        sources: dict[str, str],
        readme: str | None = None,
    ) -> Tree:
        cfg = self.cfg
        paths = sorted(parsed)

        # ---- layer 0: symbol leaves
        leaves_by_file: dict[str, list[Leaf]] = {}
        for p in paths:
            leaves_by_file[p] = make_leaves(parsed[p], sources[p], self.tok, cfg.chunking)
        leaf_nodes: dict[str, list[Node]] = {}
        for p in paths:
            nodes = []
            for leaf in leaves_by_file[p]:
                meta = leaf.metadata()
                part = f"#{leaf.part[0]}" if leaf.part else ""
                meta["key"] = f"sym:{p}::{leaf.symbol}{part}@{leaf.start_line}"
                nodes.append(self._new_node(leaf.text, 0, set(), meta))
            leaf_nodes[p] = nodes
        all_leaves = [n for p in paths for n in leaf_nodes[p]]
        self._embed(all_leaves, 0)

        if cfg.summary.depth == "symbol":
            self._symbol_summaries(all_leaves, parsed)

        # ---- layer 1: classes
        class_nodes: dict[str, dict[str, Node]] = defaultdict(dict)
        class_jobs: list[tuple[str, str, list[Node]]] = []
        for p in paths:
            groups: dict[str, list[Node]] = defaultdict(list)
            for n in leaf_nodes[p]:
                if n.metadata.get("class_name"):
                    groups[n.metadata["class_name"]].append(n)
            for cls_name, members in groups.items():
                if len(members) >= 2:
                    class_jobs.append((p, cls_name, members))
        self._build_classes(class_jobs, parsed, class_nodes)

        # ---- layer 2: files
        file_nodes = self._build_files(paths, parsed, sources, leaf_nodes, class_nodes)

        # ---- layers 3+: directories, then the repo
        dir_nodes, repo_node = self._build_dirs_and_repo(repo_name, file_nodes, readme)

        # ---- concept layer
        concept_nodes = []
        if cfg.concepts.enabled:
            members = [n for p in paths for n in class_nodes[p].values()] + [
                file_nodes[p] for p in paths if p in file_nodes
            ]
            concept_nodes = self._build_concepts(members, repo_node.layer + 1)

        layer_to_nodes: dict[int, list[Node]] = defaultdict(list)
        for n in self._nodes:
            layer_to_nodes[n.layer].append(n)
        all_nodes = {n.index: n for n in self._nodes}
        top = max(layer_to_nodes)
        roots = {repo_node.index: repo_node}
        roots.update({n.index: n for n in concept_nodes})
        tree = Tree(
            all_nodes=all_nodes,
            root_nodes=roots,
            leaf_nodes={n.index: n for n in all_leaves},
            num_layers=top,
            layer_to_nodes=dict(sorted(layer_to_nodes.items())),
            metadata={
                "kind": "code",
                "repo": repo_name,
                "embedding_key": cfg.embedding_key,
                "embedding_model": self.emb.model_id,
                "summarization_model": self.summ.model_id,
                "summary_depth": cfg.summary.depth,
                "concept_pool": self.concept_pool,
                "layers": {
                    "symbol": 0,
                    "class": 1,
                    "file": 2,
                    "dir": [3, repo_node.layer - 1] if dir_nodes else None,
                    "repo": repo_node.layer,
                    "concept": repo_node.layer + 1 if concept_nodes else None,
                },
            },
        )
        tree.validate()
        self.progress(
            {"stage": "done", "num_nodes": len(tree.all_nodes), "num_layers": tree.num_layers}
        )
        return tree

    # ------------------------------------------------------------- per level
    def _symbol_summaries(self, leaves: list[Node], parsed: dict[str, ParsedFile]) -> None:
        targets = [
            n for n in leaves if n.metadata["symbol_kind"] in ("function", "method", "merged")
        ]
        ps = [
            self._fmt_prompt(
                prompts.SYMBOL_PROMPT,
                language=n.metadata["language"],
                symbol_kind=n.metadata["symbol_kind"],
                path=n.metadata["path"],
                code=self._truncate(n.text, self.cfg.summary.file_context_tokens),
            )
            for n in targets
        ]
        summaries = self._summarize_many(ps, 0, "symbol")
        for n, p, s in zip(targets, ps, summaries, strict=True):
            n.metadata["summary"] = s
            n.metadata["input_hash"] = short_hash(p)

    def _build_classes(
        self,
        jobs: list[tuple[str, str, list[Node]]],
        parsed: dict[str, ParsedFile],
        out: dict[str, dict[str, Node]],
    ) -> None:
        symbol_depth = self.cfg.summary.depth == "symbol"
        texts: list[str] = []
        metas: list[dict[str, Any]] = []
        ps: list[str] = []
        for path, cls_name, members in jobs:
            sym = next((s for s in parsed[path].symbols if s.qualname == cls_name), None)
            header = next((m for m in members if m.metadata["symbol_kind"] == "class"), None)
            header_code = header.text if header else (sym.signature if sym else f"class {cls_name}")
            outline = []
            for m in members:
                if m.metadata["symbol_kind"] == "class":
                    continue
                label = m.metadata.get("signature") or m.metadata["symbol"]
                desc = (
                    m.metadata.get("summary")
                    if symbol_depth
                    else _first_line(m.metadata.get("docstring", ""))
                )
                outline.append((label, desc or ""))
            meta = {
                "kind": "class",
                "path": path,
                "symbol": cls_name,
                "start_line": sym.start_line
                if sym
                else min(m.metadata["start_line"] for m in members),
                "end_line": sym.end_line if sym else max(m.metadata["end_line"] for m in members),
                "signature": sym.signature if sym else "",
                "docstring": sym.docstring if sym else "",
                "bases": sym.bases if sym else [],
                "language": parsed[path].language,
                "key": f"class:{path}::{cls_name}",
            }
            if symbol_depth:
                members_txt = self._fit_children(outline, self.cfg.summary.group_context_tokens)
                ps.append(
                    self._fmt_prompt(
                        prompts.CLASS_PROMPT,
                        language=parsed[path].language,
                        name=cls_name,
                        path=path,
                        header=self._truncate(header_code, 800),
                        members=members_txt,
                    )
                )
            else:
                # File depth: an extractive outline (no LLM call).
                lines = [f"{meta['signature'] or 'class ' + cls_name}"]
                if meta["docstring"]:
                    lines.append(self._truncate(str(meta["docstring"]), 300))
                lines.append("Methods:")
                lines.extend(f"- {lbl}" + (f": {d}" if d else "") for lbl, d in outline)
                texts.append("\n".join(lines))
            metas.append(meta)
        if symbol_depth:
            summaries = self._summarize_many(ps, 1, "class")
            for meta, p, s in zip(metas, ps, summaries, strict=True):
                meta["summary"] = s
                meta["input_hash"] = short_hash(p)
                texts.append(s)
        nodes = []
        for (path, cls_name, members), meta, body in zip(jobs, metas, texts, strict=True):
            if not symbol_depth:
                meta["summary"] = body
            text = f"Class {cls_name} in {path}\n{body}"
            node = self._new_node(text, 1, {m.index for m in members}, meta)
            out[path][cls_name] = node
            nodes.append(node)
        self._embed(nodes, 1)

    def _file_body(self, path: str, pf: ParsedFile, source: str, children: list[Node]) -> str:
        parts = []
        if pf.module_docstring:
            parts.append("Module docstring:\n" + self._truncate(pf.module_docstring, 400))
        if pf.imports:
            imports = list(dict.fromkeys(i.text for i in pf.imports))
            parts.append("Imports:\n" + self._truncate("\n".join(imports), 300))
        if self.cfg.summary.depth == "symbol":
            items = []
            for c in children:
                m = c.metadata
                if m["kind"] == "class":
                    items.append((f"class {m['symbol']}", m.get("summary", "")))
                elif m.get("summary"):
                    items.append((m.get("signature") or m["symbol"], m["summary"]))
                elif m.get("symbol_kind") == "module":
                    items.append(("module-level code", self._truncate(c.text, 150)))
            parts.append(
                "Contents:\n" + self._fit_children(items, self.cfg.summary.file_context_tokens)
            )
        else:
            outline = []
            for s in pf.symbols:
                indent = "  " if s.parent else ""
                doc = _first_line(s.docstring)
                outline.append(f"{indent}{s.signature}" + (f"  # {doc}" if doc else ""))
            if outline:
                parts.append("Outline:\n" + self._truncate("\n".join(outline), 800))
            parts.append(
                "Source:\n```\n"
                + self._truncate(source, self.cfg.summary.file_context_tokens)
                + "\n```"
            )
        return "\n\n".join(parts)

    def _build_files(
        self,
        paths: list[str],
        parsed: dict[str, ParsedFile],
        sources: dict[str, str],
        leaf_nodes: dict[str, list[Node]],
        class_nodes: dict[str, dict[str, Node]],
    ) -> dict[str, Node]:
        jobs: list[tuple[str, list[Node], str | None, str | None]] = []
        for p in paths:
            if not leaf_nodes[p]:
                continue
            classes = class_nodes.get(p, {})
            children: list[Node] = list(classes.values())
            children += [n for n in leaf_nodes[p] if n.metadata.get("class_name") not in classes]
            children.sort(key=lambda n: n.metadata.get("start_line", 0))
            if self._ntok(sources[p]) < TINY_FILE_TOKENS:
                jobs.append((p, children, None, sources[p].strip()))
            else:
                body = self._file_body(p, parsed[p], sources[p], children)
                jobs.append(
                    (p, children, self._fmt_prompt(prompts.FILE_PROMPT, path=p, body=body), None)
                )
        llm = [j for j in jobs if j[2] is not None]
        summaries = iter(self._summarize_many([j[2] for j in llm if j[2]], 2, "file"))
        out: dict[str, Node] = {}
        for p, children, prompt, extractive in jobs:
            summary = next(summaries) if prompt is not None else f"Small file:\n{extractive}"
            pf = parsed[p]
            meta: dict[str, Any] = {
                "kind": "file",
                "path": p,
                "symbol": p,
                "start_line": 1,
                "end_line": max(1, pf.num_lines),
                "language": pf.language,
                "summary": summary,
                "key": f"file:{p}",
            }
            if prompt is not None:
                meta["input_hash"] = short_hash(prompt)
            out[p] = self._new_node(f"File {p}\n{summary}", 2, {c.index for c in children}, meta)
        self._embed(list(out.values()), 2)
        return out

    def _build_dirs_and_repo(
        self, repo_name: str, file_nodes: dict[str, Node], readme: str | None
    ) -> tuple[dict[str, Node], Node]:
        children_of: dict[str, list[tuple[str, str]]] = defaultdict(list)  # dir -> [(kind, name)]
        dirs: set[str] = set()
        for p in file_nodes:
            d = posixpath.dirname(p)
            children_of[d].append(("file", p))
            while d:
                parent = posixpath.dirname(d)
                dirs.add(d)
                if ("dir", d) not in children_of[parent]:
                    children_of[parent].append(("dir", d))
                d = parent

        height: dict[str, int] = {}

        def h(d: str) -> int:
            if d not in height:
                subs = [n for k, n in children_of[d] if k == "dir"]
                height[d] = 1 + max((h(s) for s in subs), default=-1)
            return height[d]

        for d in dirs:
            h(d)
        dir_nodes: dict[str, Node] = {}
        max_h = max(height.values(), default=-1)
        for level in range(max_h + 1):
            batch = sorted(d for d in dirs if height[d] == level)
            self._summarize_groups(batch, children_of, file_nodes, dir_nodes, 3 + level, kind="dir")
        repo_layer = 3 + max_h + 1
        readme_txt = (
            f"README excerpt:\n{self._truncate(readme, README_TOKENS)}\n\n" if readme else ""
        )
        self._summarize_groups(
            [""],
            children_of,
            file_nodes,
            dir_nodes,
            repo_layer,
            kind="repo",
            repo_name=repo_name,
            readme=readme_txt,
        )
        repo = dir_nodes.pop("")
        return dir_nodes, repo

    def _summarize_groups(
        self,
        names: list[str],
        children_of: dict[str, list[tuple[str, str]]],
        file_nodes: dict[str, Node],
        dir_nodes: dict[str, Node],
        layer: int,
        *,
        kind: str,
        repo_name: str = "",
        readme: str = "",
    ) -> None:
        jobs: list[tuple[str, list[Node], str | None]] = []
        for d in names:
            kids = [file_nodes[n] if k == "file" else dir_nodes[n] for k, n in children_of[d]]
            kids.sort(key=lambda n: (n.metadata["kind"] != "dir", n.metadata["path"]))
            if kind == "dir" and len(kids) == 1:
                jobs.append((d, kids, None))  # a single child: reuse its summary
                continue
            items = [
                (
                    f"{'directory' if c.metadata['kind'] == 'dir' else 'file'} `{c.metadata['path']}`",
                    c.metadata.get("summary", ""),
                )
                for c in kids
            ]
            listing = self._fit_children(items, self.cfg.summary.group_context_tokens)
            if kind == "dir":
                prompt = self._fmt_prompt(prompts.DIR_PROMPT, path=d, children=listing)
            else:
                prompt = self._fmt_prompt(
                    prompts.REPO_PROMPT, name=repo_name, readme=readme, children=listing
                )
            jobs.append((d, kids, prompt))
        summaries = iter(
            self._summarize_many([p for _, _, p in jobs if p is not None], layer, kind)
        )
        new = []
        for d, kids, job_prompt in jobs:
            summary = (
                str(kids[0].metadata.get("summary", "")) if job_prompt is None else next(summaries)
            )
            meta: dict[str, Any] = {
                "kind": kind,
                "path": d if kind == "dir" else ".",
                "symbol": d if kind == "dir" else repo_name,
                "start_line": 0,
                "end_line": 0,
                "summary": summary,
                "key": f"dir:{d}" if kind == "dir" else "repo",
            }
            if job_prompt is not None:
                meta["input_hash"] = short_hash(job_prompt)
            label = f"Directory {d}/" if kind == "dir" else f"Repository {repo_name}"
            node = self._new_node(f"{label}\n{summary}", layer, {k.index for k in kids}, meta)
            dir_nodes[d] = node
            new.append(node)
        self._embed(new, layer)

    def _build_concepts(self, members: list[Node], layer: int) -> list[Node]:
        cc = self.cfg.concepts
        if len(members) < cc.min_nodes:
            return []
        key = self.cfg.embedding_key
        by_key = {m.metadata["key"]: i for i, m in enumerate(members)}
        self.concept_pool = sorted(by_key)
        emb = np.stack([np.asarray(m.embeddings[key], np.float32) for m in members])
        clusters: list[list[int]]
        if self.reuse.concepts:
            clusters = self._reuse_concepts(by_key, emb)
        else:
            self.progress({"stage": "clustering", "layer": layer, "total": len(members)})
            tokens = [self._ntok(m.metadata.get("summary", m.text)) for m in members]
            clusterer = make_clusterer(cc.clustering)
            clusters = clusterer.cluster(emb, tokens)
        clusters = [c for c in clusters if len(c) >= cc.min_members and len(c) < len(members)]
        if not clusters:
            return []
        ps = []
        for c in clusters:
            group = sorted((members[i] for i in c), key=lambda n: n.metadata["key"])
            items = [
                (
                    f"{g.metadata['kind']} `{g.metadata['path']}"
                    + (f"::{g.metadata['symbol']}`" if g.metadata["kind"] == "class" else "`"),
                    g.metadata.get("summary", ""),
                )
                for g in group
            ]
            ps.append(
                self._fmt_prompt(
                    prompts.CONCEPT_PROMPT,
                    children=self._fit_children(items, self.cfg.summary.group_context_tokens),
                )
            )
        summaries = self._summarize_many(ps, layer, "concept")
        nodes = []
        for i, (c, p, s) in enumerate(zip(clusters, ps, summaries, strict=True)):
            name = _concept_name(s) or f"concept {i + 1}"
            paths = sorted({members[j].metadata["path"] for j in c})
            meta = {
                "kind": "concept",
                "path": paths[0] if len(paths) == 1 else posixpath.commonpath(paths) or ".",
                "paths": paths,
                "symbol": name,
                "start_line": 0,
                "end_line": 0,
                "summary": s,
                "input_hash": short_hash(p),
                "key": f"concept:{i}",
            }
            nodes.append(
                self._new_node(f"Concept: {name}\n{s}", layer, {members[j].index for j in c}, meta)
            )
        self._embed(nodes, layer)
        return nodes

    def _reuse_concepts(self, by_key: dict[str, int], emb: np.ndarray) -> list[list[int]]:
        """Keep the previous concept memberships; attach new nodes to the nearest concept."""
        assert self.reuse.concepts is not None
        clusters = [[by_key[k] for k in keys if k in by_key] for keys in self.reuse.concepts]
        clusters = [c for c in clusters if c]
        assigned = {i for c in clusters for i in c}
        prev_keys = self.reuse.concept_pool | {k for keys in self.reuse.concepts for k in keys}
        if not clusters:
            return []
        unit = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
        centroids = np.stack([unit[c].mean(axis=0) for c in clusters])
        for k, i in by_key.items():
            if i in assigned or k in prev_keys:
                continue
            # Only nodes that are new since the last clustering are attached.
            best = int(np.argmax(centroids @ unit[i]))
            clusters[best].append(i)
        return clusters


def _concept_name(summary: str) -> str | None:
    m = re.search(r"Concept:\s*\**\s*([^\n*]+)", summary)
    return m.group(1).strip().rstrip(".") if m else None
