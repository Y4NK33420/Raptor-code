"""Run QuALITY experiments from YAML configs.

An experiment file is a :class:`ExperimentConfig`: a :class:`~raptor.config.RaptorConfig`
under ``raptor:`` plus eval settings (split, method, top-k / budget overrides, embedding
shorthand). Each config hashes to a run id; results go to ``artifacts/runs/<run_id>/``::

    config.yaml     the experiment config (as run)
    results.jsonl   one row per question (appended as questions finish)
    state.json      resumable progress: cost counters, per-story tree stats and build time
    summary.json    written when every question is answered

Trees are cached under ``artifacts/trees/<article_id>/<fingerprint>/``, keyed by the
sections that shape a tree (chunking, clustering, builder, models), so retrieval-only
variants never rebuild. Every model call goes through the disk cache, so an interrupted
run resumes where it stopped and a finished config can be re-run for free.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..builder import TreeBuilder
from ..chunking import default_tokenizer, make_chunker
from ..config import RaptorConfig
from ..models import ModelBundle, build_models
from ..models.cache import CachedEmbeddingModel, CachedSummarizationModel, DiskCache, open_cache
from ..retrieval import TreeRetriever
from ..retrieval.hybrid import Reranker
from ..settings import get_settings
from ..storage import load_tree, save_tree
from ..types import Node, Tree
from .answering import PROMPT_VERSION, MCQAnswerer
from .datasets.quality import QualityArticle
from .metrics import mean_tree_stats, summarize_results, tree_stats
from .splits import DEFAULT_SPLITS_FILE, load_split_articles

logger = logging.getLogger(__name__)

DEFAULT_RUNS_DIR = Path("artifacts/runs")
DEFAULT_TREES_DIR = Path("artifacts/trees")
TREE_SECTIONS = ("chunking", "clustering", "builder", "models")
LEAF_SECTIONS = ("chunking", "models")


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    stage: str = ""  # sweep stage: "A", "B", "C"
    factor: str = (
        ""  # what this run varies: baseline|chunking|clustering|retrieval|embedding|method
    )
    baseline: str | None = None  # name of the run to compare against in reports
    method: Literal["raptor", "flat"] = "raptor"
    split: str = "dev"
    top_k: int | None = Field(None, ge=1)  # overrides raptor.retrieval.top_k
    max_tokens: int | None = Field(None, ge=1)  # overrides raptor.retrieval.max_tokens
    # Shorthand "provider:model", e.g. "local:bge-small" or "gemini:gemini-embedding-001".
    embedding: str | None = None
    raptor: RaptorConfig = Field(default_factory=RaptorConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

    def effective(self) -> RaptorConfig:
        """The RaptorConfig with this experiment's overrides applied."""
        cfg = self.raptor.model_copy(deep=True)
        if self.top_k is not None:
            cfg.retrieval.top_k = self.top_k
        if self.max_tokens is not None:
            cfg.retrieval.max_tokens = self.max_tokens
        if self.embedding:
            provider, _, model = self.embedding.partition(":")
            if not model or provider not in ("gemini", "local", "ollama"):
                raise ValueError(
                    f"embedding must look like 'local:bge-small', got {self.embedding!r}"
                )
            cfg.models.embedding_provider = provider  # type: ignore[assignment]
            cfg.models.embedding_model = model
        return cfg

    def run_id(self) -> str:
        """``<slug>-<hash>``: the hash covers everything that can change the results."""
        blob = json.dumps(
            {
                "method": self.method,
                "split": self.split,
                "config": self.effective().model_dump(mode="json"),
                "prompt": PROMPT_VERSION,
            },
            sort_keys=True,
        ).encode()
        slug = re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")[:48] or "run"
        return f"{slug}-{hashlib.sha256(blob).hexdigest()[:8]}"

    def factors(self) -> dict[str, str]:
        return factor_labels(self.effective(), self.method)


def factor_labels(cfg: RaptorConfig, method: str = "raptor") -> dict[str, str]:
    """Short human labels for the four sweep factors."""
    ch = cfg.chunking
    chunking = f"{ch.strategy}-{ch.max_tokens}" + (
        f"-o{ch.overlap_tokens}" if ch.overlap_tokens else ""
    )
    cl = cfg.clustering
    clustering = cl.strategy if method == "raptor" else "none"
    if cl.params and method == "raptor":
        clustering += "(" + ",".join(f"{k}={v}" for k, v in sorted(cl.params.items())) + ")"
    r = cfg.retrieval
    retrieval = (r.mode if method == "raptor" else "flat") + f"-k{r.top_k}"
    if r.max_tokens != 3500:
        retrieval += f"-t{r.max_tokens}"
    if r.hybrid:
        retrieval += "+bm25"
    if r.reranker:
        retrieval += "+rerank"
    if r.mmr_lambda is not None:
        retrieval += f"+mmr{r.mmr_lambda:g}"
    m = cfg.models
    embedding = m.embedding_model.split("/")[-1]
    if not m.use_task_types and (m.embedding_provider or m.provider) == "gemini":
        embedding += "-notask"
    return {
        "chunking": chunking,
        "clustering": clustering,
        "retrieval": retrieval,
        "embedding": embedding,
    }


# --------------------------------------------------------------------- tree cache
@dataclass
class TreeRecord:
    tree: Tree
    build_seconds: float  # time of the original build (0 for cache-only rebuilds)
    built_now: bool
    stats: dict[str, float]


class TreeStore:
    """Trees on disk, keyed by article and the tree-shaping part of the config."""

    def __init__(self, root: str | Path = DEFAULT_TREES_DIR) -> None:
        self.root = Path(root)
        self._locks: dict[str, threading.Lock] = {}

    @staticmethod
    def key(cfg: RaptorConfig, method: str) -> str:
        if method == "flat":
            return "leaves-" + cfg.fingerprint(sections=LEAF_SECTIONS)
        return cfg.fingerprint(sections=TREE_SECTIONS)

    def path(self, article_id: str, cfg: RaptorConfig, method: str) -> Path:
        return self.root / article_id / self.key(cfg, method)

    def get_or_build(
        self, article: QualityArticle, cfg: RaptorConfig, method: str, models: ModelBundle
    ) -> TreeRecord:
        path = self.path(article.article_id, cfg, method)
        lock = self._locks.setdefault(str(path), threading.Lock())
        with lock:
            meta_file = path / "build.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                return TreeRecord(load_tree(path), float(meta["seconds"]), False, meta["stats"])
            t0 = time.perf_counter()
            if method == "flat":
                tree = build_leaf_tree(article.article, cfg, models)
            else:
                tree = TreeBuilder(cfg, models.embedding, models.summarization).build_from_text(
                    article.article
                )
            seconds = time.perf_counter() - t0
            tok = default_tokenizer()
            stats = tree_stats(tree, lambda t: len(tok.encode(t, disallowed_special=())))
            save_tree(tree, path)
            meta = {
                "article_id": article.article_id,
                "title": article.title,
                "method": method,
                "seconds": seconds,
                "stats": stats,
                "created_at": _now(),
                "config": {s: cfg.model_dump(mode="json")[s] for s in TREE_SECTIONS},
            }
            meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            logger.info(
                "built %s tree for %s in %.1fs: %d nodes, %d layers",
                method,
                article.article_id,
                seconds,
                len(tree.all_nodes),
                tree.num_layers + 1,
            )
            return TreeRecord(tree, seconds, True, stats)


def build_leaf_tree(text: str, cfg: RaptorConfig, models: ModelBundle) -> Tree:
    """Chunk and embed only (the flat-RAG index): a tree with just layer 0."""
    chunks = make_chunker(cfg.chunking, models.embedding).split(text)
    emb = models.embedding.embed_documents([c.text for c in chunks])
    key = cfg.builder.embedding_key
    leaves = [
        Node(c.text, i, set(), {key: emb[i]}, 0, {"start": c.start, "end": c.end})
        for i, c in enumerate(chunks)
    ]
    return Tree.from_layers(
        {0: leaves},
        metadata={"embedding_key": key, "embedding_model": models.embedding.model_id},
    )


# ------------------------------------------------------------------------- runner
_BUNDLES: dict[str, ModelBundle] = {}
_BUNDLES_LOCK = threading.Lock()


def shared_models(cfg: RaptorConfig) -> ModelBundle:
    """One model bundle per models config per process (local models load once)."""
    key = json.dumps(cfg.models.model_dump(mode="json"), sort_keys=True)
    with _BUNDLES_LOCK:
        if key not in _BUNDLES:
            _BUNDLES[key] = build_models(cfg.models)
        return _BUNDLES[key]


@dataclass
class Counters:
    """Snapshot of cache counters, to measure one run's cost as a difference."""

    emb_hits: int = 0
    emb_misses: int = 0
    sum_hits: int = 0
    sum_misses: int = 0
    qa_hits: int = 0
    qa_misses: int = 0

    @classmethod
    def read(cls, models: ModelBundle, answerer: MCQAnswerer) -> Counters:
        emb = models.embedding.stats if isinstance(models.embedding, CachedEmbeddingModel) else None
        summ = (
            models.summarization.stats
            if isinstance(models.summarization, CachedSummarizationModel)
            else None
        )
        return cls(
            emb.hits if emb else 0,
            emb.misses if emb else 0,
            summ.hits if summ else 0,
            summ.misses if summ else 0,
            answerer.stats.hits,
            answerer.stats.misses,
        )

    def minus(self, other: Counters) -> Counters:
        return Counters(*(a - b for a, b in zip(self.as_tuple(), other.as_tuple(), strict=True)))

    def as_tuple(self) -> tuple[int, ...]:
        return (
            self.emb_hits,
            self.emb_misses,
            self.sum_hits,
            self.sum_misses,
            self.qa_hits,
            self.qa_misses,
        )

    def cost(self) -> dict[str, int]:
        return {
            "llm_calls": self.sum_misses + self.qa_misses,
            "embed_calls": self.emb_misses,
            "cache_hits": self.emb_hits + self.sum_hits + self.qa_hits,
            "cache_misses": self.emb_misses + self.sum_misses + self.qa_misses,
            "summary_calls": self.sum_misses,
            "answer_calls": self.qa_misses,
        }


@dataclass
class RunOutcome:
    run_id: str
    name: str
    status: str  # "done" | "skipped" | "quota"
    summary: dict[str, Any] | None = None
    session_cost: dict[str, int] = field(default_factory=dict)


class Runner:
    def __init__(
        self,
        runs_dir: str | Path = DEFAULT_RUNS_DIR,
        trees_dir: str | Path = DEFAULT_TREES_DIR,
        *,
        splits_file: str | Path = DEFAULT_SPLITS_FILE,
        data_dir: str | Path | None = None,
        models_factory: Callable[[RaptorConfig], ModelBundle] = shared_models,
        cache: DiskCache | None = None,
        reranker: Reranker | None = None,
        question_workers: int = 4,
        articles_loader: Callable[[str], list[QualityArticle]] | None = None,
    ) -> None:
        self.runs_dir = Path(runs_dir)
        self.trees = TreeStore(trees_dir)
        self.splits_file = Path(splits_file)
        self.data_dir = data_dir
        self.models_factory = models_factory
        self._cache = cache
        self.reranker = reranker
        self.question_workers = question_workers
        self._articles_loader = articles_loader
        self._answerers: dict[str, MCQAnswerer] = {}

    # ------------------------------------------------------------------ helpers
    def cache(self) -> DiskCache | None:
        if self._cache is None and get_settings().cache_enabled:
            self._cache = open_cache(get_settings().cache_dir)
        return self._cache

    def articles(self, split: str) -> list[QualityArticle]:
        if self._articles_loader is not None:
            return self._articles_loader(split)
        kwargs: dict[str, Any] = {"splits_file": self.splits_file}
        if self.data_dir is not None:
            kwargs["data_dir"] = self.data_dir
        return load_split_articles(split, **kwargs)

    def answerer(self, models: ModelBundle) -> MCQAnswerer:
        key = models.qa.model_id
        if key not in self._answerers:
            self._answerers[key] = MCQAnswerer.from_qa_model(models.qa, self.cache())
        return self._answerers[key]

    # --------------------------------------------------------------------- run
    def run_file(self, path: str | Path, *, force: bool = False) -> RunOutcome:
        return self.run(ExperimentConfig.from_yaml(path), force=force)

    def run(self, exp: ExperimentConfig, *, force: bool = False) -> RunOutcome:
        run_id = exp.run_id()
        run_dir = self.runs_dir / run_id
        summary_file = run_dir / "summary.json"
        if summary_file.exists() and not force:
            logger.info("%s: already finished (use --force to redo)", run_id)
            return RunOutcome(
                run_id, exp.name, "skipped", json.loads(summary_file.read_text("utf-8"))
            )
        if force and run_dir.exists():
            for f in ("results.jsonl", "state.json", "summary.json"):
                (run_dir / f).unlink(missing_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(exp.to_yaml(), encoding="utf-8")

        cfg = exp.effective()
        models = self.models_factory(cfg)
        answerer = self.answerer(models)
        results_file = run_dir / "results.jsonl"
        previous = read_jsonl(results_file)
        if previous:  # rewrite cleanly: drops a torn last line left by an interruption
            results_file.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in previous),
                encoding="utf-8",
            )
        done = {r["question_id"] for r in previous}
        state = _load_state(run_dir)
        start = Counters.read(models, answerer)
        write_lock = threading.Lock()
        tok = default_tokenizer()
        status = "done"

        def checkpoint() -> None:
            delta = Counters.read(models, answerer).minus(start)
            cost = {k: state["cost_before"].get(k, 0) + v for k, v in delta.cost().items()}
            state["cost"] = cost
            (run_dir / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

        from ..models import keypool

        keypool.FAIL_ON_DAILY_QUOTA = True
        try:
            for article in self.articles(exp.split):
                pending = [q for q in article.questions if q.question_id not in done]
                record = self.trees.get_or_build(article, cfg, exp.method, models)
                state["trees"][article.article_id] = {
                    "build_seconds": record.build_seconds,
                    "stats": record.stats,
                }
                if not pending:
                    continue
                retriever = TreeRetriever(
                    record.tree, models.embedding, cfg.retrieval, reranker=self.reranker
                )
                mode = "collapsed" if exp.method == "flat" else cfg.retrieval.mode

                def answer_one(
                    q: Any,
                    retriever: TreeRetriever = retriever,
                    art: QualityArticle = article,
                    mode: str = mode,
                ) -> None:
                    t0 = time.perf_counter()
                    res = retriever.retrieve(q.question, mode=mode)
                    context = res.context
                    ans = answerer.answer(context, q.question, q.options)
                    row = {
                        "run_id": run_id,
                        "set_id": art.set_id,
                        "article_id": art.article_id,
                        "question_id": q.question_id,
                        "question": q.question,
                        "options": q.options,
                        "gold": q.gold_label,
                        "predicted": ans.option,
                        "correct": ans.option is not None and ans.option == q.gold_label,
                        "difficult": q.difficult,
                        "retrieved": [
                            {
                                "node_index": r.node.index,
                                "layer": r.layer,
                                "score": round(r.score, 5),
                            }
                            for r in res.nodes
                        ],
                        "context_tokens": len(tok.encode(context, disallowed_special=())),
                        "latency_s": round(time.perf_counter() - t0, 4),
                        "rationale": ans.rationale,
                        "parsed_by": ans.parsed_by,
                    }
                    with write_lock, results_file.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

                with ThreadPoolExecutor(max_workers=self.question_workers) as pool:
                    for fut in [pool.submit(answer_one, q) for q in pending]:
                        fut.result()
                checkpoint()
        except keypool.QuotaExhaustedError as exc:
            logger.error("%s: stopping, %s. Progress is saved; re-run to resume.", run_id, exc)
            status = "quota"
        finally:
            keypool.FAIL_ON_DAILY_QUOTA = False
            checkpoint()

        session = Counters.read(models, answerer).minus(start).cost()
        if status != "done":
            return RunOutcome(run_id, exp.name, status, None, session)
        summary = build_summary(exp, run_id, read_jsonl(results_file), state)
        summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info(
            "%s: accuracy %.3f on %d questions; this session %d cache hits, %d misses "
            "(%d LLM calls, %d embeddings)",
            run_id,
            summary["accuracy"],
            summary["n_questions"],
            session["cache_hits"],
            session["cache_misses"],
            session["llm_calls"],
            session["embed_calls"],
        )
        return RunOutcome(run_id, exp.name, "done", summary, session)

    def run_many(self, paths: Sequence[str | Path], *, force: bool = False) -> list[RunOutcome]:
        outcomes = []
        for p in paths:
            out = self.run_file(p, force=force)
            outcomes.append(out)
            if out.status == "quota":
                logger.error("daily quota exhausted; not starting the remaining configs")
                break
        return outcomes


# ------------------------------------------------------------------- artifacts
def build_summary(
    exp: ExperimentConfig, run_id: str, rows: list[dict[str, Any]], state: dict[str, Any]
) -> dict[str, Any]:
    cfg = exp.effective()
    trees = list(state.get("trees", {}).values())
    acc = summarize_results(rows)
    build = [t["build_seconds"] for t in trees]
    return {
        "run_id": run_id,
        "name": exp.name,
        "description": exp.description,
        "stage": exp.stage,
        "factor": exp.factor,
        "baseline": exp.baseline,
        "created_at": state.get("created_at", _now()),
        "finished_at": _now(),
        "split": exp.split,
        "method": exp.method,
        "config_fingerprint": cfg.fingerprint(),
        "tree_fingerprint": TreeStore.key(cfg, exp.method),
        "factors": exp.factors(),
        "n_questions": acc["n_questions"],
        "accuracy": acc["accuracy"],
        "accuracy_hard": acc["accuracy_hard"],
        "n_hard": acc["n_hard"],
        "ci95": acc["ci95"],
        "layer_share": acc["layer_share"],
        "context_tokens_mean": acc["context_tokens_mean"],
        "cost": {
            k: state.get("cost", {}).get(k, 0)
            for k in (
                "llm_calls",
                "embed_calls",
                "cache_hits",
                "cache_misses",
                "summary_calls",
                "answer_calls",
            )
        },
        "build_seconds": float(sum(build) / len(build)) if build else 0.0,
        "query_seconds_mean": acc["query_seconds_mean"],
        "tree_stats": mean_tree_stats([t["stats"] for t in trees]),
        "n_stories": len(trees),
        "reader_model": cfg.models.llm_model,
        "prompt_version": PROMPT_VERSION,
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:  # a torn last line from an interrupted run
                    logger.warning("skipping unreadable line in %s", path)
    # Keep the last row per question (a resumed run never duplicates, but be safe).
    return list({r["question_id"]: r for r in rows}.values())


def _load_state(run_dir: Path) -> dict[str, Any]:
    f = run_dir / "state.json"
    state = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    state.setdefault("created_at", _now())
    state.setdefault("trees", {})
    state.setdefault("cost", {})
    state["cost_before"] = dict(state["cost"])
    return state


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
