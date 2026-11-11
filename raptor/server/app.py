"""FastAPI application factory for the RAPTOR web UI."""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..chunking import default_tokenizer
from ..config import ModelsConfig, RaptorConfig
from ..models import ModelBundle, build_models
from ..types import Node, nodes_text
from . import schemas as S
from .builds import PRESETS, preset_config, run_build
from .code import HAVE_CODE, CodeService
from .config import ServerSettings
from .fakes import fake_bundle
from .indexes import IndexRegistry, SearcherCache, preview
from .jobs import JobManager
from .projection import cached_projection
from .quality import QualityStore
from .runs import RunStore
from .runs import _eval_runs as _phase2_runs
from .sse import Event, sse_response

logger = logging.getLogger(__name__)
_OPTION = re.compile(r"\boption\s*\(?(\d+)\)?", re.I)


def parse_option(answer: str, n_options: int) -> int | None:
    m = _OPTION.search(answer[:400])
    if m and 1 <= int(m.group(1)) <= n_options:
        return int(m.group(1))
    return None


class _Models:
    """Injected, fake, or lazily created real models (shared by every request)."""

    def __init__(self, settings: ServerSettings, injected: ModelBundle | None) -> None:
        self.injected = injected
        self.settings = settings
        self._bundle = injected or (
            fake_bundle(settings.fake_delay) if settings.fake_models else None
        )
        self._lock = threading.Lock()

    @property
    def offline(self) -> bool:
        return self.injected is not None or self.settings.fake_models

    def __call__(self) -> ModelBundle:
        with self._lock:
            if self._bundle is None:
                self._bundle = build_models(ModelsConfig())
            return self._bundle

    def for_build(self, config: RaptorConfig) -> ModelBundle:
        return self() if self.offline else build_models(config.models)


def create_app(
    settings: ServerSettings | None = None, *, models: ModelBundle | None = None
) -> FastAPI:
    """Create the API (and, when ``settings.web_dist`` exists, the static frontend).

    ``models`` overrides every model (tests pass deterministic fakes here).
    """
    settings = settings or ServerSettings()
    mdl = _Models(settings, models)
    quality = QualityStore(settings.quality_dir, settings.quality_split)
    registry = IndexRegistry(
        settings.demo_dir, settings.indexes_dir, quality_lookup=quality.by_position
    )
    searchers = SearcherCache(registry, mdl, fake=mdl.offline)
    jobs = JobManager()
    runs = RunStore(settings.runs_dirs)
    code = CodeService(settings.code_dir, settings.code_stub_repos, mdl)
    tokenizer = default_tokenizer()
    token_cache: dict[tuple[str, int], int] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        jobs.shutdown()

    app = FastAPI(
        lifespan=lifespan,
        title="RAPTOR",
        version=__version__,
        description="Build, explore and query RAPTOR indexes; browse experiments and code.",
    )
    app.state.settings = settings
    app.state.jobs = jobs
    app.state.registry = registry
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    api = APIRouter(prefix="/api")

    def not_found(what: str) -> HTTPException:
        return HTTPException(status_code=404, detail=f"{what} not found")

    def index_or_404(index_id: str) -> dict[str, Any]:
        try:
            return registry.info(index_id)
        except KeyError:
            raise not_found(f"index {index_id!r}") from None

    def ntokens(index_id: str, node: Node) -> int:
        key = (index_id, node.index)
        if key not in token_cache:
            token_cache[key] = len(tokenizer.encode(node.text, disallowed_special=()))
        return token_cache[key]

    # ------------------------------------------------------------------- meta
    @api.get("/info", response_model=S.ServerInfo, tags=["meta"])
    def info() -> dict[str, Any]:
        cfg = ModelsConfig()
        bundle_ids = ("fake", "fake") if mdl.offline else (cfg.llm_model, cfg.embedding_model)
        if mdl.injected is not None:
            bundle_ids = (mdl.injected.qa.model_id, mdl.injected.embedding.model_id)
        return {
            "version": __version__,
            "fake_models": mdl.offline,
            "code_backend": "raptor.code" if code.real else "stub",
            "runs_backend": "raptor.eval.runs" if _phase2_runs is not None else "files",
            "quality_available": quality.available,
            "llm_model": bundle_ids[0],
            "embedding_model": bundle_ids[1],
        }

    @api.get("/meta/events", response_model=S.EventCatalog, tags=["meta"], include_in_schema=True)
    def event_catalog() -> None:
        """Documents the SSE payload shapes (for client type generation); not meant to be called."""
        raise HTTPException(status_code=204)

    @api.get("/presets", response_model=list[S.Preset], tags=["indexes"])
    def presets() -> list[dict[str, Any]]:
        return [
            {
                "id": k,
                "label": v["label"],
                "description": v["description"],
                "config": preset_config(k).model_dump(mode="json"),
            }
            for k, v in PRESETS.items()
        ]

    # ---------------------------------------------------------------- indexes
    @api.get("/indexes", response_model=list[S.IndexInfo], tags=["indexes"])
    def list_indexes() -> list[dict[str, Any]]:
        out = []
        for e in registry.entries():
            try:
                out.append(registry.info(e.id))
            except Exception as exc:
                logger.warning("skipping unreadable index %s: %s", e.path, exc)
        return sorted(out, key=lambda i: (i["origin"] != "user", i["created_at"]), reverse=False)

    @api.get("/indexes/{index_id}", response_model=S.IndexInfo, tags=["indexes"])
    def get_index(index_id: str) -> dict[str, Any]:
        return index_or_404(index_id)

    @api.delete("/indexes/{index_id}", status_code=204, tags=["indexes"])
    def delete_index(index_id: str) -> None:
        index_or_404(index_id)
        try:
            registry.delete(index_id)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from None

    @api.post("/indexes", response_model=S.BuildAccepted, status_code=202, tags=["indexes"])
    async def create_index(
        source_type: Annotated[Literal["text", "file", "quality"], Form()],
        name: Annotated[str | None, Form(max_length=120)] = None,
        text: Annotated[str | None, Form()] = None,
        set_id: Annotated[str | None, Form()] = None,
        preset: Annotated[str, Form()] = "balanced",
        config: Annotated[str | None, Form(description="JSON RaptorConfig overrides")] = None,
        file: Annotated[UploadFile | None, File()] = None,
    ) -> dict[str, Any]:
        source: dict[str, Any] = {"type": source_type}
        if source_type == "text":
            if not text or not text.strip():
                raise HTTPException(422, "text is empty")
            body, default_name = text, "Pasted text"
        elif source_type == "file":
            if file is None:
                raise HTTPException(422, "no file uploaded")
            if not (file.filename or "").lower().endswith((".txt", ".md", ".markdown")):
                raise HTTPException(415, "only .txt and .md files are supported")
            raw = await file.read(settings.max_upload_bytes + 1)
            if len(raw) > settings.max_upload_bytes:
                raise HTTPException(413, "file is too large")
            body = raw.decode("utf-8", errors="replace")
            default_name = Path(file.filename or "upload").stem
            source["filename"] = file.filename
        else:
            try:
                art = quality.get(set_id or "")
            except KeyError:
                raise not_found(f"QuALITY story {set_id!r}") from None
            body, default_name = art.article, art.title
            source.update(set_id=art.set_id, title=art.title)
        if len(body.strip()) < 20:
            raise HTTPException(422, "the document is too short to index")
        try:
            overrides = json.loads(config) if config else None
            cfg = preset_config(preset, overrides)
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, f"invalid config: {exc}") from None
        display = (name or "").strip() or default_name
        index_id, out_dir = registry.new_path(display, uuid.uuid4().hex[:6])
        bundle = mdl.for_build(cfg)
        job = jobs.submit(
            "build",
            lambda job: run_build(
                job,
                text=body,
                name=display,
                source=source,
                preset=preset,
                config=cfg,
                models=bundle,
                index_id=index_id,
                out_dir=out_dir,
            ),
            meta={"index_id": index_id, "name": display, "preset": preset, "source": source},
        )
        return {"job_id": job.id, "index_id": index_id}

    # ------------------------------------------------------------------- jobs
    @api.get("/jobs", response_model=list[S.JobInfo], tags=["jobs"])
    def list_jobs() -> list[dict[str, Any]]:
        return [j.summary() for j in jobs.all()]

    @api.get("/jobs/{job_id}", response_model=S.JobInfo, tags=["jobs"])
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            return jobs.get(job_id).summary()
        except KeyError:
            raise not_found("job") from None

    @api.get("/jobs/{job_id}/events", tags=["jobs"], response_class=JSONResponse)
    def job_events(
        job_id: str,
        after: int = -1,
        last_event_id: Annotated[str | None, Header()] = None,
    ):
        """Server-sent events: ``queued``, ``start``, ``progress`` (TreeBuilder events),
        ``chunks``, ``clusters``, ``summary``, then ``complete`` or ``error``."""
        try:
            job = jobs.get(job_id)
        except KeyError:
            raise not_found("job") from None
        if last_event_id and last_event_id.isdigit():
            after = int(last_event_id)
        return sse_response(job.stream(after))

    # ------------------------------------------------------------------- tree
    @api.get("/indexes/{index_id}/tree", response_model=S.TreeData, tags=["tree"])
    def get_tree(index_id: str) -> dict[str, Any]:
        info_ = index_or_404(index_id)
        tree = registry.tree(index_id)
        parents: dict[int, list[int]] = {}
        edges = []
        for n in tree.all_nodes.values():
            for c in sorted(n.children):
                parents.setdefault(c, []).append(n.index)
                edges.append({"source": n.index, "target": c})
        nodes = [
            {
                "id": n.index,
                "layer": n.layer,
                "preview": preview(n.text, 160),
                "children": sorted(n.children),
                "parents": sorted(parents.get(n.index, [])),
                "tokens": ntokens(index_id, n),
                "metadata": n.metadata,
            }
            for n in (tree.all_nodes[i] for i in sorted(tree.all_nodes))
        ]
        return {
            "id": index_id,
            "num_layers": tree.num_layers,
            "layer_counts": info_["layer_counts"],
            "nodes": nodes,
            "edges": edges,
        }

    @api.get("/indexes/{index_id}/nodes/{nid}", response_model=S.NodeDetail, tags=["tree"])
    def get_node(index_id: str, nid: int) -> dict[str, Any]:
        index_or_404(index_id)
        tree = registry.tree(index_id)
        if nid not in tree.all_nodes:
            raise not_found(f"node {nid}")
        node = tree.all_nodes[nid]
        parents = tree.parents_of(nid)
        siblings = sorted({c for p in parents for c in tree.all_nodes[p].children} - {nid})
        return {
            "id": nid,
            "layer": node.layer,
            "text": node.text,
            "tokens": ntokens(index_id, node),
            "children": sorted(node.children),
            "parents": sorted(parents),
            "siblings": siblings,
            "metadata": node.metadata,
        }

    @api.get("/indexes/{index_id}/projection", response_model=S.Projection, tags=["tree"])
    def get_projection(index_id: str) -> dict[str, Any]:
        index_or_404(index_id)
        entry = registry.entry(index_id)
        tree = registry.tree(index_id)
        key = f"{index_id}-{int(entry.mtime)}"
        return cached_projection(tree, settings.derived_dir, key, settings.projection_method)

    # ------------------------------------------------------------------ query
    def _context(picked: list[tuple[Node, float]]) -> str:
        return nodes_text(n for n, _ in picked)

    @api.post("/indexes/{index_id}/query", tags=["query"], response_class=JSONResponse)
    def query(index_id: str, req: S.QueryRequest):
        """SSE: ``retrieval`` (nodes, layers, scores), ``token``…, ``answer``, ``done``."""
        index_or_404(index_id)

        def events() -> Iterator[Event]:
            t0 = time.time()
            searcher = searchers.get(index_id)
            res = searcher.raptor(
                req.question, top_k=req.top_k, max_tokens=req.max_tokens, mode=req.mode
            )
            picked = [(r.node, r.score) for r in res.nodes]
            yield (
                "retrieval",
                {**searcher.describe(picked), "latency_s": round(time.time() - t0, 3)},
            )
            parts: list[str] = []
            for piece in mdl().qa.stream_answer(_context(picked), req.question):
                parts.append(piece)
                yield "token", {"text": piece}
            yield "answer", {"answer": "".join(parts), "latency_s": round(time.time() - t0, 3)}
            yield "done", {}

        return sse_response(events())

    @api.post("/indexes/{index_id}/compare", tags=["query"], response_class=JSONResponse)
    def compare(index_id: str, req: S.CompareRequest):
        """SSE: ``question``, then interleaved ``retrieval``/``token``/``answer`` events tagged
        with ``side`` ("flat" or "raptor"), then ``done``. Both sides get the same top-k and
        token budget."""
        index_or_404(index_id)
        options: list[str] | None = None
        gold: int | None = None
        meta: dict[str, Any] = {}
        if req.question_id:
            try:
                art, q = quality.question(req.question_id, req.set_id)
            except KeyError:
                raise not_found(f"question {req.question_id!r}") from None
            question, options, gold = q.question, q.options, q.gold_label
            prompt = q.mcq_prompt()
            meta = {"question_id": q.question_id, "set_id": art.set_id, "difficult": q.difficult}
        elif req.question and req.question.strip():
            question = prompt = req.question
        else:
            raise HTTPException(422, "give a question or a question_id")

        def events() -> Iterator[Event]:
            yield (
                "question",
                {
                    "question": question,
                    "options": options,
                    "gold": gold,
                    **meta,
                    "budget": {"top_k": req.top_k, "max_tokens": req.max_tokens},
                },
            )
            searcher = searchers.get(index_id)
            qa = mdl().qa
            out: queue.Queue[Event | None] = queue.Queue()

            def side(name: Literal["flat", "raptor"]) -> None:
                t0 = time.time()
                try:
                    if name == "raptor":
                        res = searcher.raptor(
                            question, top_k=req.top_k, max_tokens=req.max_tokens, mode="collapsed"
                        )
                        picked = [(r.node, r.score) for r in res.nodes]
                    else:
                        picked = searcher.flat(question, top_k=req.top_k, max_tokens=req.max_tokens)
                    out.put(
                        (
                            "retrieval",
                            {
                                "side": name,
                                **searcher.describe(picked),
                                "latency_s": round(time.time() - t0, 3),
                            },
                        )
                    )
                    parts: list[str] = []
                    for piece in qa.stream_answer(_context(picked), prompt):
                        parts.append(piece)
                        out.put(("token", {"side": name, "text": piece}))
                    answer = "".join(parts)
                    predicted = parse_option(answer, len(options)) if options else None
                    out.put(
                        (
                            "answer",
                            {
                                "side": name,
                                "answer": answer,
                                "latency_s": round(time.time() - t0, 3),
                                "predicted": predicted,
                                "correct": (predicted == gold) if gold and options else None,
                            },
                        )
                    )
                except Exception as exc:
                    logger.exception("compare side %s failed", name)
                    out.put(("error", {"side": name, "message": f"{type(exc).__name__}: {exc}"}))
                finally:
                    out.put(None)

            threads = [
                threading.Thread(target=side, args=(n,), daemon=True) for n in ("flat", "raptor")
            ]
            for t in threads:
                t.start()
            remaining = len(threads)
            while remaining:
                item = out.get()
                if item is None:
                    remaining -= 1
                else:
                    yield item
            yield "done", {}

        return sse_response(events())

    # ---------------------------------------------------------------- quality
    def _story_index_ids() -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for e in registry.entries():
            try:
                src = registry.info(e.id)["source"]
            except Exception:
                continue
            if src.get("set_id"):
                out.setdefault(src["set_id"], []).append(e.id)
        return out

    @api.get("/quality/stories", response_model=list[S.QualityStory], tags=["quality"])
    def quality_stories(limit: Annotated[int, Query(ge=1, le=1000)] = 200) -> list[dict[str, Any]]:
        ids = _story_index_ids()
        return [
            {**quality.story_summary(a), "index_ids": ids.get(a.set_id, [])}
            for a in quality.articles()[:limit]
        ]

    @api.get("/quality/stories/{set_id}", response_model=S.QualityStoryDetail, tags=["quality"])
    def quality_story(set_id: str) -> dict[str, Any]:
        try:
            a = quality.get(set_id)
        except KeyError:
            raise not_found("story") from None
        return {
            **quality.story_summary(a),
            "index_ids": _story_index_ids().get(a.set_id, []),
            "article": a.article,
            "questions": [quality.question_dict(q) for q in a.questions],
        }

    # ------------------------------------------------------------------- runs
    @api.get("/runs", response_model=list[S.RunSummary], tags=["runs"])
    def list_runs() -> list[dict[str, Any]]:
        return runs.list_all()

    @api.get("/runs/compare", response_model=S.RunCompare, tags=["runs"])
    def compare_runs(a: str, b: str) -> dict[str, Any]:
        try:
            return runs.compare(a, b)
        except KeyError as exc:
            raise not_found(f"run {exc.args[0]!r}") from None

    @api.get("/runs/{run_id}", response_model=S.RunDetail, tags=["runs"])
    def get_run(run_id: str) -> dict[str, Any]:
        try:
            return runs.load(run_id)
        except KeyError:
            raise not_found("run") from None

    # ------------------------------------------------------------------- code
    def code_call(fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except KeyError as exc:
            raise not_found(str(exc.args[0]) if exc.args else "code index") from None

    @api.get("/code", response_model=list[S.CodeRepo], tags=["code"])
    def list_code() -> list[dict[str, Any]]:
        return code.list_all()

    @api.get("/code/{code_id}/files", response_model=list[S.CodeFile], tags=["code"])
    def code_files(code_id: str) -> list[dict[str, Any]]:
        return code_call(lambda: code.files(code_id))

    @api.get("/code/{code_id}/file", response_model=S.CodeFileText, tags=["code"])
    def code_file(code_id: str, path: str) -> dict[str, Any]:
        return code_call(lambda: code.file_text(code_id, path))

    @api.get("/code/{code_id}/symbol/{nid}", response_model=S.CodeSymbol, tags=["code"])
    def code_symbol(code_id: str, nid: int) -> dict[str, Any]:
        return code_call(lambda: code.symbol(code_id, nid))

    @api.get("/code/{code_id}/graph", response_model=S.CodeGraph, tags=["code"])
    def code_graph(code_id: str, kinds: str | None = None) -> dict[str, Any]:
        wanted = [k for k in (kinds or "").split(",") if k] or None
        return code_call(lambda: code.graph(code_id, wanted))

    @api.post("/code/{code_id}/ask", tags=["code"], response_class=JSONResponse)
    def code_ask(code_id: str, req: S.CodeAskRequest):
        """SSE: ``token``… then ``citations`` (list of code hits) and ``done``."""
        idx = code_call(lambda: code.get(code_id))

        def events() -> Iterator[Event]:
            from .code import as_dict

            t0 = time.time()
            parts: list[str] = []
            for piece in idx.stream_ask(req.question, req.k):
                if isinstance(piece, str):
                    parts.append(piece)
                    yield "token", {"text": piece}
                else:
                    cites = [as_dict(c) for c in getattr(piece, "citations", [])]
                    yield "citations", {"citations": cites}
                    yield (
                        "answer",
                        {
                            "answer": getattr(piece, "answer", "".join(parts)),
                            "latency_s": round(time.time() - t0, 3),
                        },
                    )
            yield "done", {}

        return sse_response(events())

    app.include_router(api)
    _mount_frontend(app, settings.web_dist)

    return app


def _mount_frontend(app: FastAPI, dist: Path) -> None:
    index = dist / "index.html"
    if not index.is_file():

        @app.get("/", include_in_schema=False)
        def no_frontend() -> JSONResponse:
            return JSONResponse(
                {
                    "message": "The web UI is not built. Run `npm install && npm run build` in web/, "
                    "or use `npm run dev` there. The API is at /api and its docs at /docs."
                }
            )

        return
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(404, "not found")
        candidate = (dist / path).resolve()
        if path and candidate.is_file() and dist.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(index)


def app_from_env() -> FastAPI:
    """Factory used by ``raptor serve --dev`` (uvicorn reload needs an import string)."""
    import os

    raw = os.environ.get("RAPTOR_SERVER_SETTINGS")
    settings = ServerSettings.model_validate_json(raw) if raw else ServerSettings()
    return create_app(settings)


__all__ = ["HAVE_CODE", "create_app", "parse_option"]
