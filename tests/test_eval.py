"""Experiment harness: splits, answering, metrics, runner, runs API and report (offline)."""

from __future__ import annotations

import json
import re

import pytest
from conftest import EchoSummarizer, HashEmbedder, make_document

from raptor.eval.answering import MCQAnswerer, parse_answer
from raptor.eval.datasets.quality import QualityArticle, QualityQuestion
from raptor.eval.metrics import bootstrap_ci, layer_share, mcnemar_exact, paired_comparison
from raptor.eval.runner import ExperimentConfig, Runner, read_jsonl
from raptor.eval.runs import compare_runs, find_run, list_runs, load_run
from raptor.eval.splits import load_splits, make_splits, save_splits
from raptor.models import ModelBundle, with_cache
from raptor.models.base import BaseQAModel
from raptor.models.cache import DiskCache


# ------------------------------------------------------------------ answering
@pytest.mark.parametrize(
    ("raw", "expected", "how"),
    [
        ('{"rationale": "because", "option": 3}', 3, "json"),
        ('```json\n{"option": 2, "rationale": "x"}\n```', 2, "json"),
        ('{"option": 9, "rationale": "out of range"} so Option 4', 4, "regex"),
        ("Option 1 is right because...", 1, "regex"),
        ("**Option 2**", 2, "regex"),
        ("I think the answer is (4).", 4, "regex"),
        ("no idea", None, "none"),
    ],
)
def test_parse_answer(raw, expected, how):
    option, _, parsed_by = parse_answer(raw, 4)
    assert option == expected and parsed_by == how


def test_answerer_caches_replies(tmp_path):
    calls = []

    def gen(prompt: str) -> str:
        calls.append(prompt)
        return '{"option": 2, "rationale": "r"}'

    cache = DiskCache(tmp_path / "c.sqlite")
    a = MCQAnswerer(gen, "fake", cache)
    first = a.answer("ctx", "q?", ["a", "b", "c", "d"])
    again = MCQAnswerer(gen, "fake", cache).answer("ctx", "q?", ["a", "b", "c", "d"])
    assert first.option == again.option == 2
    assert not first.cached and again.cached and len(calls) == 1
    assert "1. a" in calls[0] and "Question: q?" in calls[0]


# -------------------------------------------------------------------- metrics
def test_mcnemar_and_ci():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 2**10)
    assert mcnemar_exact(5, 5) == 1.0
    lo, hi = bootstrap_ci([True] * 7 + [False] * 3)
    assert lo < 0.7 < hi
    cmp = paired_comparison({"a": True, "b": False, "c": False}, {"a": True, "b": True, "c": False})
    assert cmp["fixed"] == ["b"] and cmp["broken"] == [] and cmp["delta"] == pytest.approx(1 / 3)
    assert cmp["both_right"] == 1 and cmp["both_wrong"] == 1
    assert layer_share([[{"layer": 0}, {"layer": 1}], [{"layer": 0}]]) == {"0": 2 / 3, "1": 1 / 3}


# --------------------------------------------------------------------- splits
def _articles(n_articles: int = 20) -> list[QualityArticle]:
    out = []
    for a in range(n_articles):
        for s in range(2):
            out.append(
                QualityArticle(
                    set_id=f"{a}_{s}",
                    article_id=str(a),
                    title="t",
                    author="x",
                    article="text",
                    questions=[QualityQuestion(f"{a}_{s}_q", "q", ["a", "b"], 1)],
                )
            )
    return out


def test_splits_are_seeded_and_disjoint(tmp_path):
    s1 = make_splits(_articles(), seed=1, size=4)
    s2 = make_splits(_articles(), seed=1, size=4)
    assert s1["dev"].set_ids == s2["dev"].set_ids
    dev_articles = {s.split("_")[0] for s in s1["dev"].set_ids}
    confirm_articles = {s.split("_")[0] for s in s1["confirm"].set_ids}
    assert len(dev_articles) == 4 and not dev_articles & confirm_articles
    path = save_splits(s1, tmp_path / "splits.yaml", seed=1)
    assert load_splits(path)["confirm"].set_ids == s1["confirm"].set_ids


def test_committed_splits_file():
    splits = load_splits("configs/splits.yaml")
    assert len(splits["dev"].set_ids) == 8 and len(splits["confirm"].set_ids) == 8
    articles = lambda ids: {s.split("_")[0] for s in ids}  # noqa: E731
    assert not articles(splits["dev"].set_ids) & articles(splits["confirm"].set_ids)


# --------------------------------------------------------------------- runner
class KeywordQA(BaseQAModel):
    """JSON-mode reader: picks the option whose words appear most in the context."""

    model_id = "test:keyword-qa"

    def __init__(self) -> None:
        self.calls = 0

    def answer_question(self, context: str, question: str) -> str:  # pragma: no cover
        raise AssertionError("runner should use JSON mode")

    def generate_json(self, prompt: str, schema: dict) -> str:
        self.calls += 1
        context = prompt.split("Question:")[0].lower()
        options = re.findall(r"^(\d)\. (.+)$", prompt, re.MULTILINE)
        best = max(options, key=lambda o: sum(w in context for w in o[1].lower().split()))
        return json.dumps({"rationale": "keyword overlap", "option": int(best[0])})


def _story_articles(split: str) -> list[QualityArticle]:
    doc = make_document(sentences_per_topic=12, seed=0 if split == "dev" else 1)
    qs = [
        QualityQuestion(
            f"{split}-q{i}", f"What happens with the {t}?", opts, gold, difficult=i % 2 == 0
        )
        for i, (t, opts, gold) in enumerate(
            [
                ("rocket", ["kitchen flour", "rocket orbit planet", "deer owl", "reef"], 2),
                ("baker", ["oven bread baker", "comet", "ship", "pine"], 1),
                ("sailor", ["moss", "astronaut", "sailor ship harbor", "dough"], 3),
                ("owl", ["galaxy", "salt", "spice", "owl forest trees"], 4),
            ]
        )
    ]
    return [
        QualityArticle(f"{split}_s1", f"art-{split}", "Topics", "Anon", doc, questions=qs[:2]),
        QualityArticle(f"{split}_s2", f"art-{split}", "Topics", "Anon", doc, questions=qs[2:]),
    ]


@pytest.fixture
def harness(tmp_path):
    cache = DiskCache(tmp_path / "cache.sqlite")
    counters = {"emb": HashEmbedder(), "sum": EchoSummarizer(), "qa": KeywordQA()}

    def factory(cfg):
        return with_cache(ModelBundle(counters["emb"], counters["sum"], counters["qa"]), cache)

    def make_runner():
        return Runner(
            tmp_path / "runs",
            tmp_path / "trees",
            models_factory=factory,
            cache=cache,
            articles_loader=_story_articles,
            question_workers=2,
        )

    return make_runner, counters, tmp_path


def _exp(name: str = "base", **kw) -> ExperimentConfig:
    data = {
        "name": name,
        "stage": "A",
        "raptor": {
            "chunking": {"max_tokens": 40},
            "clustering": {"strategy": "contiguous"},
            "retrieval": {"top_k": 4},
            "models": {"provider": "gemini"},
        },
    }
    data.update(kw)
    return ExperimentConfig.model_validate(data)


SUMMARY_KEYS = {
    "run_id", "name", "created_at", "split", "method", "config_fingerprint", "factors",
    "n_questions", "accuracy", "accuracy_hard", "ci95", "layer_share", "cost",
    "build_seconds", "query_seconds_mean", "tree_stats",
}  # fmt: skip
ROW_KEYS = {
    "run_id", "set_id", "question_id", "question", "options", "gold", "predicted", "correct",
    "difficult", "retrieved", "context_tokens", "latency_s", "rationale",
}  # fmt: skip


def test_runner_writes_contract_artifacts(harness):
    make_runner, counters, tmp = harness
    exp = _exp()
    out = make_runner().run(exp)
    assert out.status == "done"
    run_dir = tmp / "runs" / out.run_id
    assert (run_dir / "config.yaml").exists()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert set(summary) >= SUMMARY_KEYS
    assert set(summary["factors"]) == {"chunking", "clustering", "retrieval", "embedding"}
    assert set(summary["cost"]) >= {"llm_calls", "embed_calls", "cache_hits", "cache_misses"}
    assert set(summary["tree_stats"]) >= {"mean_layers", "mean_nodes", "mean_branching"}
    assert summary["method"] == "raptor" and summary["n_questions"] == 4
    assert summary["accuracy"] == 1.0  # keyword reader + topical retrieval
    rows = read_jsonl(run_dir / "results.jsonl")
    assert all(set(r) >= ROW_KEYS for r in rows)
    assert all({"node_index", "layer", "score"} == set(r["retrieved"][0]) for r in rows)
    # One tree per article, built once even though the article has two question sets.
    assert len(list((tmp / "trees").glob("*/*/tree.json"))) == 1
    assert summary["cost"]["answer_calls"] == 4 and counters["qa"].calls == 4


def test_rerun_makes_zero_model_calls(harness):
    make_runner, counters, tmp = harness
    exp = _exp()
    first = make_runner().run(exp)
    calls = (counters["emb"].texts_embedded, counters["sum"].calls, counters["qa"].calls)
    assert make_runner().run(exp).status == "skipped"
    again = make_runner().run(exp, force=True)
    assert again.status == "done"
    assert again.session_cost["cache_misses"] == 0 and again.session_cost["llm_calls"] == 0
    assert (counters["emb"].texts_embedded, counters["sum"].calls, counters["qa"].calls) == calls
    assert again.summary is not None and first.summary is not None
    assert again.summary["accuracy"] == first.summary["accuracy"]
    assert len(read_jsonl(tmp / "runs" / first.run_id / "results.jsonl")) == 4


def test_retrieval_only_variant_reuses_tree(harness):
    make_runner, counters, _tmp = harness
    runner = make_runner()
    runner.run(_exp())
    summaries_before = counters["sum"].calls
    variant = _exp(
        "hybrid", raptor={**_exp().raptor.model_dump(), "retrieval": {"top_k": 4, "hybrid": True}}
    )
    out = runner.run(variant)
    assert out.status == "done" and counters["sum"].calls == summaries_before
    assert out.summary["factors"]["retrieval"] == "collapsed-k4+bm25"
    assert out.run_id != runner.run(_exp()).run_id


def test_flat_method_uses_leaf_only_tree(harness):
    make_runner, counters, _tmp = harness
    out = make_runner().run(_exp("flat", method="flat"))
    assert out.summary["method"] == "flat" and counters["sum"].calls == 0
    assert out.summary["layer_share"] == {"0": 1.0}
    assert out.summary["factors"]["retrieval"].startswith("flat-k4")


def test_resume_after_interruption(harness):
    make_runner, _counters, tmp = harness
    exp = _exp()
    out = make_runner().run(exp)
    run_dir = tmp / "runs" / out.run_id
    lines = (run_dir / "results.jsonl").read_text().splitlines()
    (run_dir / "results.jsonl").write_text("\n".join(lines[:2]) + "\n" + lines[2][:10])  # torn line
    (run_dir / "summary.json").unlink()
    resumed = make_runner().run(exp)
    assert resumed.status == "done" and resumed.summary["n_questions"] == 4
    assert resumed.session_cost["answer_calls"] == 0  # the missing answers come from cache


def test_runs_api_and_report(harness):
    make_runner, _, tmp = harness
    runner = make_runner()
    base = runner.run(_exp())
    flat = runner.run(_exp("flat", method="flat", baseline="base", factor="method"))
    root = tmp / "runs"
    assert {r["run_id"] for r in list_runs(root)} == {base.run_id, flat.run_id}
    summary, rows = load_run(base.run_id, root)
    assert summary["run_id"] == base.run_id and len(rows) == 4
    cmp = compare_runs(base.run_id, flat.run_id, root)
    assert set(cmp) >= {"delta", "mcnemar_p", "fixed", "broken", "both_right", "both_wrong"}
    assert find_run("flat", root)["run_id"] == flat.run_id

    from raptor.eval.report import generate_report

    path = generate_report(root, tmp / "report")
    text = path.read_text(encoding="utf-8")
    assert "Stage A" in text and "`flat`" in text and "McNemar" in text
    assert (tmp / "report" / "figures" / "stage_a.png").exists()


def test_experiment_config_embedding_shorthand():
    exp = _exp(embedding="local:bge-small")
    cfg = exp.effective()
    assert cfg.models.embedding_provider == "local" and cfg.models.embedding_model == "bge-small"
    assert exp.factors()["embedding"] == "bge-small"
    with pytest.raises(ValueError):
        _exp(embedding="bge-small").effective()
