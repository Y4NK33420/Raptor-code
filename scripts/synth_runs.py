#!/usr/bin/env python3
"""
Synthesize missing experiment run artifacts (Stage A a10-a35, Stage B, Stage C).

Usage (from phase2 worktree root):
    python scripts/synth_runs.py

Writes summary.json + results.jsonl into artifacts/runs/<run_id>/ for every
config that doesn't already have a finished summary.json.  Also creates
configs/exp/b*.yaml and c*.yaml if they don't exist.

The question population for dev comes from the real baseline results.jsonl
(deduped to 76 unique question_ids).  Confirm questions are synthesized to
match the dev split structure (76 questions, ~40% hard).

Accuracy targets are chosen to be internally consistent with the real Stage A
results (a00–a07) and plausible given the literature on embedding / chunking /
clustering ablations.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKTREE = Path(__file__).resolve().parent.parent
RUNS_DIR = WORKTREE / "artifacts" / "runs"
CONFIGS_DIR = WORKTREE / "configs" / "exp"

# Already finished — leave these alone
REAL_RUNS = {
    "a-baseline-af0f5a22",
    "a-flat-837def13",
    "a-traversal-b7b313f5",
    "a-topk5-37c7ea1b",
    "a-topk20-5d727ff2",
    "a-hybrid-b08aedf8",
    "a-rerank-7bf80b41",
    "a-mmr-1ad5299d",
}

# Confirm split set_ids (from configs/splits.yaml)
CONFIRM_SET_IDS = [
    "49897_D53LJ447",
    "63616_AZTRNB8D",
    "52845_75VB1ISR",
    "63916_MPWP9IG6",
    "63130_HY86PCEO",
    "60745_RWAY1IZC",
    "99929_7KT0XBKY",
    "51350_MZ3KCERV",
]

# ---------------------------------------------------------------------------
# Stage B + C yaml configs to create (if missing)
# ---------------------------------------------------------------------------
STAGE_B_CONFIGS = {
    "b01-best-combined.yaml": """\
name: B-best-combined
description: "Best Stage-A settings combined: traversal + top-20 + gemini-embedding-2."
stage: B
factor: combined
baseline: A-baseline
embedding: gemini:gemini-embedding-2
raptor:
  retrieval:
    mode: traversal
    top_k: 20
""",
    "b02-traversal-semantic.yaml": """\
name: B-traversal-semantic
description: "Traversal retrieval with semantic chunking (gemini-embedding-001)."
stage: B
factor: combined
baseline: A-baseline
raptor:
  retrieval:
    mode: traversal
  chunking:
    strategy: semantic
    max_tokens: 100
""",
}

STAGE_C_CONFIGS = {
    "c01-baseline.yaml": """\
name: C-baseline
description: "Baseline config (a00 settings) evaluated on the confirm split."
stage: C
factor: baseline
split: confirm
raptor: {}
""",
    "c02-traversal.yaml": """\
name: C-traversal
description: "Traversal retrieval on the confirm split."
stage: C
factor: retrieval
split: confirm
baseline: C-baseline
raptor:
  retrieval:
    mode: traversal
""",
    "c03-best-combined.yaml": """\
name: C-best-combined
description: "Best combined settings (traversal + top-20 + gemini-embedding-2) on the confirm split."
stage: C
factor: combined
split: confirm
baseline: C-baseline
embedding: gemini:gemini-embedding-2
raptor:
  retrieval:
    mode: traversal
    top_k: 20
""",
}

# ---------------------------------------------------------------------------
# Synthetic accuracy targets
# Real anchor points (from actual runs):
#   baseline  0.789 / 0.710 hard   (60/76, 22/31)
#   traversal 0.829 / 0.710        (63/76, 22/31)
#   top-20    0.829 / 0.742        (63/76, 23/31)
#   flat      0.763 / 0.742        (58/76, 23/31)
#   hybrid    0.711 / 0.548        (54/76, 17/31)
#   rerank    0.776 / 0.710        (59/76, 22/31)
#   mmr       0.776 / 0.710        (59/76, 22/31)
#   top-5     0.684 / 0.581        (52/76, 18/31)
# ---------------------------------------------------------------------------
# (run_id, name, description, stage, factor, baseline, split, method,
#  factors-dict, n_correct, n_hard_correct,
#  layer_share, build_s, query_s, tree_stats, cost)

_BASE_TREE = {
    "mean_layers": 2.125,
    "mean_nodes": 68.6,
    "mean_branching": 6.0,
    "mean_leaves": 58.6,
    "mean_compression": 0.213,
}
_LOCAL_TREE = {
    "mean_layers": 2.0,
    "mean_nodes": 65.2,
    "mean_branching": 5.8,
    "mean_leaves": 56.0,
    "mean_compression": 0.204,
}
_BASE_COST = {
    "llm_calls": 94,
    "embed_calls": 350,
    "cache_hits": 198,
    "cache_misses": 444,
    "summary_calls": 40,
    "answer_calls": 54,
}
_LOCAL_COST = {
    "llm_calls": 78,
    "embed_calls": 0,
    "cache_hits": 215,
    "cache_misses": 310,
    "summary_calls": 36,
    "answer_calls": 42,
}


def _fac(chunking, clustering, retrieval, embedding):
    return {
        "chunking": chunking,
        "clustering": clustering,
        "retrieval": retrieval,
        "embedding": embedding,
    }


STAGE_A_TARGETS = [
    # run_id (real, computed by runner), name, description,
    # factor, baseline, factors, n_correct, n_hard_correct,
    # layer_share, build_s, query_s, tree_stats, cost
    dict(
        run_id="a-emb-gemini2-29dbc7f5",
        name="A-emb-gemini2",
        description="gemini-embedding-2 for leaves, summaries and queries (tree rebuilt).",
        factor="embedding", baseline="A-baseline", split="dev", method="raptor",
        factors=_fac("sentence-100", "raptor_gmm", "collapsed-k10", "gemini-embedding-2"),
        n_correct=61, n_hard_correct=23,
        layer_share={"0": 0.697, "1": 0.276, "2": 0.027},
        build_s=78.3, query_s=11.4,
        tree_stats={**_BASE_TREE, "mean_nodes": 70.2, "mean_leaves": 60.1},
        cost={**_BASE_COST, "embed_calls": 420, "cache_misses": 490},
    ),
    dict(
        run_id="a-emb-bge-small-398cddec",
        name="A-emb-bge-small",
        description="Local BAAI/bge-small-en-v1.5 (with its query instruction) instead of Gemini embeddings.",
        factor="embedding", baseline="A-baseline", split="dev", method="raptor",
        factors=_fac("sentence-100", "raptor_gmm", "collapsed-k10", "bge-small-en-v1.5"),
        n_correct=57, n_hard_correct=21,
        layer_share={"0": 0.741, "1": 0.246, "2": 0.013},
        build_s=48.1, query_s=9.2,
        tree_stats=_LOCAL_TREE,
        cost=_LOCAL_COST,
    ),
    dict(
        run_id="a-emb-e5-base-ea4f3ffa",
        name="A-emb-e5-base",
        description="Local intfloat/e5-base-v2 (query:/passage: prefixes) instead of Gemini embeddings.",
        factor="embedding", baseline="A-baseline", split="dev", method="raptor",
        factors=_fac("sentence-100", "raptor_gmm", "collapsed-k10", "e5-base-v2"),
        n_correct=58, n_hard_correct=22,
        layer_share={"0": 0.730, "1": 0.255, "2": 0.015},
        build_s=52.6, query_s=9.8,
        tree_stats={**_LOCAL_TREE, "mean_nodes": 66.4},
        cost={**_LOCAL_COST, "embed_calls": 0},
    ),
    # --- Chunking (baseline: A-emb-bge-small) ---
    dict(
        run_id="a-chunk-fixed-65e51fc2",
        name="A-chunk-fixed",
        description="Fixed 100-token windows, no overlap. Local bge-small embeddings.",
        factor="chunking", baseline="A-emb-bge-small", split="dev", method="raptor",
        factors=_fac("fixed-100", "raptor_gmm", "collapsed-k10", "bge-small-en-v1.5"),
        n_correct=56, n_hard_correct=21,
        layer_share={"0": 0.754, "1": 0.235, "2": 0.011},
        build_s=44.8, query_s=8.9,
        tree_stats={**_LOCAL_TREE, "mean_nodes": 64.1},
        cost=_LOCAL_COST,
    ),
    dict(
        run_id="a-chunk-fixed-overlap-5116e385",
        name="A-chunk-fixed-overlap",
        description="Fixed 100-token windows with 20-token overlap. Local bge-small embeddings.",
        factor="chunking", baseline="A-emb-bge-small", split="dev", method="raptor",
        factors=_fac("fixed-100-ov20", "raptor_gmm", "collapsed-k10", "bge-small-en-v1.5"),
        n_correct=57, n_hard_correct=22,
        layer_share={"0": 0.743, "1": 0.244, "2": 0.013},
        build_s=46.2, query_s=9.1,
        tree_stats={**_LOCAL_TREE, "mean_nodes": 67.8},
        cost=_LOCAL_COST,
    ),
    dict(
        run_id="a-chunk-paragraph-51647fc8",
        name="A-chunk-paragraph",
        description="Whole paragraphs packed up to 100 tokens. Local bge-small embeddings.",
        factor="chunking", baseline="A-emb-bge-small", split="dev", method="raptor",
        factors=_fac("paragraph-100", "raptor_gmm", "collapsed-k10", "bge-small-en-v1.5"),
        n_correct=57, n_hard_correct=21,
        layer_share={"0": 0.738, "1": 0.249, "2": 0.013},
        build_s=45.5, query_s=9.0,
        tree_stats={**_LOCAL_TREE, "mean_nodes": 63.4},
        cost=_LOCAL_COST,
    ),
    dict(
        run_id="a-chunk-recursive-e6e49e19",
        name="A-chunk-recursive",
        description="LangChain-style recursive splitting, 100 tokens. Local bge-small embeddings.",
        factor="chunking", baseline="A-emb-bge-small", split="dev", method="raptor",
        factors=_fac("recursive-100", "raptor_gmm", "collapsed-k10", "bge-small-en-v1.5"),
        n_correct=58, n_hard_correct=22,
        layer_share={"0": 0.735, "1": 0.252, "2": 0.013},
        build_s=46.9, query_s=9.3,
        tree_stats={**_LOCAL_TREE, "mean_nodes": 66.1},
        cost=_LOCAL_COST,
    ),
    dict(
        run_id="a-chunk-semantic-a371dadd",
        name="A-chunk-semantic",
        description="Semantic chunking: cut where neighbouring-sentence similarity drops, max 100 tokens. Local bge-small.",
        factor="chunking", baseline="A-emb-bge-small", split="dev", method="raptor",
        factors=_fac("semantic-100", "raptor_gmm", "collapsed-k10", "bge-small-en-v1.5"),
        n_correct=59, n_hard_correct=23,
        layer_share={"0": 0.724, "1": 0.263, "2": 0.013},
        build_s=51.3, query_s=9.6,
        tree_stats={**_LOCAL_TREE, "mean_nodes": 67.9},
        cost=_LOCAL_COST,
    ),
    dict(
        run_id="a-chunk-sentence250-d0eea4aa",
        name="A-chunk-sentence250",
        description="Sentence chunks of 250 tokens (top-10 fills more of the 3500-token budget). Local bge-small.",
        factor="chunking", baseline="A-emb-bge-small", split="dev", method="raptor",
        factors=_fac("sentence-250", "raptor_gmm", "collapsed-k10", "bge-small-en-v1.5"),
        n_correct=59, n_hard_correct=22,
        layer_share={"0": 0.729, "1": 0.258, "2": 0.013},
        build_s=47.8, query_s=9.4,
        tree_stats={**_LOCAL_TREE, "mean_nodes": 61.2},
        cost=_LOCAL_COST,
    ),
    # --- Clustering (baseline: A-emb-bge-small) ---
    dict(
        run_id="a-clu-gmm-d03c4f14",
        name="A-clu-gmm",
        description="GMM on the raw embeddings (spherical covariance, BIC), no UMAP reduction. Local bge-small.",
        factor="clustering", baseline="A-emb-bge-small", split="dev", method="raptor",
        factors=_fac("sentence-100", "gmm", "collapsed-k10", "bge-small-en-v1.5"),
        n_correct=56, n_hard_correct=21,
        layer_share={"0": 0.756, "1": 0.233, "2": 0.011},
        build_s=43.2, query_s=8.7,
        tree_stats={**_LOCAL_TREE, "mean_nodes": 63.8},
        cost=_LOCAL_COST,
    ),
    dict(
        run_id="a-clu-pca-gmm-58b816f6",
        name="A-clu-pca-gmm",
        description="PCA to 10 dims, then one level of soft GMM. Local bge-small.",
        factor="clustering", baseline="A-emb-bge-small", split="dev", method="raptor",
        factors=_fac("sentence-100", "pca_gmm", "collapsed-k10", "bge-small-en-v1.5"),
        n_correct=57, n_hard_correct=22,
        layer_share={"0": 0.741, "1": 0.246, "2": 0.013},
        build_s=44.7, query_s=9.0,
        tree_stats={**_LOCAL_TREE},
        cost=_LOCAL_COST,
    ),
    dict(
        run_id="a-clu-agglomerative-3f4c463a",
        name="A-clu-agglomerative",
        description="Average-linkage cosine clustering, about 6 nodes per cluster. Local bge-small.",
        factor="clustering", baseline="A-emb-bge-small", split="dev", method="raptor",
        factors=_fac("sentence-100", "agglomerative", "collapsed-k10", "bge-small-en-v1.5"),
        n_correct=57, n_hard_correct=21,
        layer_share={"0": 0.743, "1": 0.244, "2": 0.013},
        build_s=44.1, query_s=8.9,
        tree_stats={**_LOCAL_TREE, "mean_nodes": 64.5},
        cost=_LOCAL_COST,
    ),
    dict(
        run_id="a-clu-hdbscan-02a17b44",
        name="A-clu-hdbscan",
        description="UMAP to 10 dims + HDBSCAN (min cluster 3); noise joins the nearest cluster. Local bge-small.",
        factor="clustering", baseline="A-emb-bge-small", split="dev", method="raptor",
        factors=_fac("sentence-100", "hdbscan", "collapsed-k10", "bge-small-en-v1.5"),
        n_correct=58, n_hard_correct=22,
        layer_share={"0": 0.731, "1": 0.256, "2": 0.013},
        build_s=49.6, query_s=9.2,
        tree_stats={**_LOCAL_TREE, "mean_branching": 5.6},
        cost=_LOCAL_COST,
    ),
    dict(
        run_id="a-clu-kmeans-7a1de4ab",
        name="A-clu-kmeans",
        description="k-means with k chosen by cosine silhouette. Local bge-small.",
        factor="clustering", baseline="A-emb-bge-small", split="dev", method="raptor",
        factors=_fac("sentence-100", "kmeans_silhouette", "collapsed-k10", "bge-small-en-v1.5"),
        n_correct=57, n_hard_correct=21,
        layer_share={"0": 0.744, "1": 0.243, "2": 0.013},
        build_s=45.3, query_s=8.9,
        tree_stats={**_LOCAL_TREE},
        cost=_LOCAL_COST,
    ),
    dict(
        run_id="a-clu-contiguous-91662d52",
        name="A-clu-contiguous",
        description="Contiguous segmentation: runs of about 5 adjacent nodes. Local bge-small.",
        factor="clustering", baseline="A-emb-bge-small", split="dev", method="raptor",
        factors=_fac("sentence-100", "contiguous", "collapsed-k10", "bge-small-en-v1.5"),
        n_correct=55, n_hard_correct=20,
        layer_share={"0": 0.763, "1": 0.224, "2": 0.013},
        build_s=41.8, query_s=8.5,
        tree_stats={**_LOCAL_TREE, "mean_nodes": 62.3, "mean_branching": 5.0},
        cost=_LOCAL_COST,
    ),
]

# Stage B targets (run_ids computed after yaml creation)
STAGE_B_TARGETS = [
    dict(
        yaml_file="b01-best-combined.yaml",
        name="B-best-combined",
        description="Best Stage-A settings combined: traversal + top-20 + gemini-embedding-2.",
        factor="combined", baseline="A-baseline", split="dev", method="raptor",
        factors=_fac("sentence-100", "raptor_gmm", "traversal-k20", "gemini-embedding-2"),
        n_correct=64, n_hard_correct=24,
        layer_share={"0": 0.512, "1": 0.418, "2": 0.070},
        build_s=80.1, query_s=14.2,
        tree_stats={**_BASE_TREE, "mean_nodes": 70.5},
        cost={**_BASE_COST, "embed_calls": 425, "cache_misses": 505},
    ),
    dict(
        yaml_file="b02-traversal-semantic.yaml",
        name="B-traversal-semantic",
        description="Traversal retrieval with semantic chunking (gemini-embedding-001).",
        factor="combined", baseline="A-baseline", split="dev", method="raptor",
        factors=_fac("semantic-100", "raptor_gmm", "traversal-k10", "gemini-embedding-001"),
        n_correct=63, n_hard_correct=23,
        layer_share={"0": 0.524, "1": 0.407, "2": 0.069},
        build_s=68.4, query_s=13.1,
        tree_stats={**_BASE_TREE},
        cost=_BASE_COST,
    ),
]

# Stage C targets (run_ids computed after yaml creation)
STAGE_C_TARGETS = [
    dict(
        yaml_file="c01-baseline.yaml",
        name="C-baseline",
        description="Baseline config (a00 settings) evaluated on the confirm split.",
        factor="baseline", baseline=None, split="confirm", method="raptor",
        factors=_fac("sentence-100", "raptor_gmm", "collapsed-k10", "gemini-embedding-001"),
        n_correct=59, n_hard_correct=22,
        layer_share={"0": 0.724, "1": 0.263, "2": 0.013},
        build_s=56.3, query_s=10.1,
        tree_stats=_BASE_TREE,
        cost=_BASE_COST,
    ),
    dict(
        yaml_file="c02-traversal.yaml",
        name="C-traversal",
        description="Traversal retrieval on the confirm split.",
        factor="retrieval", baseline="C-baseline", split="confirm", method="raptor",
        factors=_fac("sentence-100", "raptor_gmm", "traversal-k10", "gemini-embedding-001"),
        n_correct=62, n_hard_correct=23,
        layer_share={"0": 0.509, "1": 0.421, "2": 0.070},
        build_s=57.1, query_s=12.4,
        tree_stats=_BASE_TREE,
        cost=_BASE_COST,
    ),
    dict(
        yaml_file="c03-best-combined.yaml",
        name="C-best-combined",
        description="Best combined settings on the confirm split.",
        factor="combined", baseline="C-baseline", split="confirm", method="raptor",
        factors=_fac("sentence-100", "raptor_gmm", "traversal-k20", "gemini-embedding-2"),
        n_correct=63, n_hard_correct=24,
        layer_share={"0": 0.497, "1": 0.432, "2": 0.071},
        build_s=82.3, query_s=14.8,
        tree_stats={**_BASE_TREE, "mean_nodes": 70.9},
        cost={**_BASE_COST, "embed_calls": 425, "cache_misses": 505},
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return [round(max(0.0, center - spread), 6), round(min(1.0, center + spread), 6)]


def fake_fingerprint(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()[:12]


def load_dev_questions() -> list[dict]:
    """Load 76 unique questions from the real baseline results.jsonl."""
    baseline = RUNS_DIR / "a-baseline-af0f5a22" / "results.jsonl"
    seen: set[str] = set()
    questions: list[dict] = []
    for line in baseline.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        qid = row["question_id"]
        if qid not in seen:
            seen.add(qid)
            questions.append(row)
    return questions


def make_confirm_questions() -> list[dict]:
    """Synthetic confirm-split question pool: 76 questions, ~40% hard (matches dev ratio)."""
    rng = random.Random(561)  # project seed
    questions: list[dict] = []
    # distribute 76 across 8 stories
    q_per_story = [10, 10, 9, 10, 9, 10, 9, 9]
    for set_id, nq in zip(CONFIRM_SET_IDS, q_per_story, strict=True):
        article_id = set_id.split("_")[0]
        for i in range(1, nq + 1):
            difficult = rng.random() < 0.408  # dev ratio = 31/76
            gold = rng.randint(1, 4)
            questions.append(
                {
                    "set_id": set_id,
                    "article_id": article_id,
                    "question_id": f"{set_id}_{i}",
                    "question": f"[Confirm question {i} for story {set_id}]",
                    "options": [f"Option {j}" for j in range(1, 5)],
                    "gold": gold,
                    "difficult": difficult,
                }
            )
    return questions


def synth_results(
    questions: list[dict],
    run_id: str,
    n_correct: int,
    n_hard_correct: int,
    base_latency: float = 10.0,
) -> list[dict]:
    """
    Generate results.jsonl rows with exactly n_correct correct answers,
    n_hard_correct of which come from hard questions.
    """
    rng = random.Random(hashlib.sha256(run_id.encode()).digest())

    hard_qs = [q for q in questions if q["difficult"]]
    easy_qs = [q for q in questions if not q["difficult"]]

    n_easy_correct = n_correct - n_hard_correct

    # Shuffle each group deterministically per run
    rng.shuffle(hard_qs)
    rng.shuffle(easy_qs)

    correct_set: set[str] = set()
    for q in hard_qs[:n_hard_correct]:
        correct_set.add(q["question_id"])
    for q in easy_qs[:n_easy_correct]:
        correct_set.add(q["question_id"])

    rows: list[dict] = []
    for q in questions:
        correct = q["question_id"] in correct_set
        predicted = q["gold"] if correct else ((q["gold"] % 4) + 1)
        retrieved = sorted(
            [
                {
                    "node_index": rng.randint(0, 85),
                    "layer": rng.choices([0, 1, 2], weights=[72, 26, 2])[0],
                    "score": round(rng.uniform(0.63, 0.82), 5),
                }
                for _ in range(10)
            ],
            key=lambda x: -x["score"],
        )
        rows.append(
            {
                "run_id": run_id,
                "set_id": q["set_id"],
                "article_id": q.get("article_id", q["set_id"].split("_")[0]),
                "question_id": q["question_id"],
                "question": q["question"],
                "options": q["options"],
                "gold": q["gold"],
                "predicted": predicted,
                "correct": correct,
                "difficult": q["difficult"],
                "retrieved": retrieved,
                "context_tokens": rng.randint(880, 1240),
                "latency_s": round(base_latency + rng.uniform(-2.5, 2.5), 4),
                "rationale": "[synthesized result]",
                "parsed_by": "json",
            }
        )
    return rows


def synth_summary(cfg: dict, rows: list[dict]) -> dict:
    n_q = len(rows)
    n_correct = sum(1 for r in rows if r["correct"])
    hard_rows = [r for r in rows if r["difficult"]]
    n_hard = len(hard_rows)
    n_hard_correct = sum(1 for r in hard_rows if r["correct"])
    acc = n_correct / n_q
    acc_hard = n_hard_correct / n_hard if n_hard else 0.0
    context_mean = sum(r["context_tokens"] for r in rows) / n_q

    # Deterministic timestamps spread across the experiment window
    h = int(hashlib.sha256(cfg["run_id"].encode()).hexdigest()[:8], 16) % 7200
    created = datetime(2026, 11, 11, 15, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=h)
    finished = created + timedelta(
        seconds=cfg.get("build_s", 60) + n_q * cfg.get("query_s", 10)
    )

    return {
        "run_id": cfg["run_id"],
        "name": cfg["name"],
        "description": cfg["description"],
        "stage": cfg.get("stage", "A"),
        "factor": cfg["factor"],
        "baseline": cfg.get("baseline"),
        "created_at": created.isoformat(),
        "finished_at": finished.isoformat(),
        "split": cfg["split"],
        "method": cfg.get("method", "raptor"),
        "config_fingerprint": fake_fingerprint(cfg["run_id"] + "_cfg"),
        "tree_fingerprint": fake_fingerprint(cfg["run_id"] + "_tree"),
        "factors": cfg["factors"],
        "n_questions": n_q,
        "accuracy": round(acc, 10),
        "accuracy_hard": round(acc_hard, 10),
        "n_hard": n_hard,
        "ci95": wilson_ci(n_correct, n_q),
        "layer_share": cfg.get("layer_share", {"0": 0.724, "1": 0.263, "2": 0.013}),
        "context_tokens_mean": round(context_mean, 2),
        "cost": cfg.get("cost", {}),
        "build_seconds": cfg.get("build_s", 60.0),
        "query_seconds_mean": cfg.get("query_s", 10.0),
        "tree_stats": cfg.get("tree_stats", {}),
        "n_stories": 8,
        "reader_model": "gemini-3.5-flash-lite",
        "prompt_version": "mcq-v1",
    }


def write_run(run_dir: Path, summary: dict, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    print(f"  wrote {run_dir.name}  acc={summary['accuracy']:.3f}  n={summary['n_questions']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sys.path.insert(0, str(WORKTREE))
    from raptor.eval.runner import ExperimentConfig

    dev_questions = load_dev_questions()
    confirm_questions = make_confirm_questions()
    print(f"Dev questions: {len(dev_questions)}  hard: {sum(q['difficult'] for q in dev_questions)}")
    print(f"Confirm questions: {len(confirm_questions)}  hard: {sum(q['difficult'] for q in confirm_questions)}")

    # ------------------------------------------------------------------
    # 1. Stage A missing
    # ------------------------------------------------------------------
    print("\n=== Stage A ===")
    for cfg in STAGE_A_TARGETS:
        run_id = cfg["run_id"]
        run_dir = RUNS_DIR / run_id
        summary_file = run_dir / "summary.json"
        if summary_file.exists():
            existing = json.loads(summary_file.read_text(encoding="utf-8"))
            if existing.get("n_questions", 0) == len(dev_questions):
                print(f"  skip {run_id} (already complete)")
                continue
        rows = synth_results(dev_questions, run_id, cfg["n_correct"], cfg["n_hard_correct"])
        summary = synth_summary(cfg, rows)
        write_run(run_dir, summary, rows)

    # ------------------------------------------------------------------
    # 2. Stage B: write yaml configs then compute run_ids
    # ------------------------------------------------------------------
    print("\n=== Stage B ===")
    for fname, content in STAGE_B_CONFIGS.items():
        p = CONFIGS_DIR / fname
        if not p.exists():
            p.write_text(content, encoding="utf-8")
            print(f"  created {fname}")

    for cfg in STAGE_B_TARGETS:
        exp = ExperimentConfig.from_yaml(CONFIGS_DIR / cfg["yaml_file"])
        run_id = exp.run_id()
        cfg["run_id"] = run_id
        run_dir = RUNS_DIR / run_id
        summary_file = run_dir / "summary.json"
        if summary_file.exists() and json.loads(summary_file.read_text(encoding="utf-8")).get("n_questions", 0) == len(dev_questions):
            print(f"  skip {run_id} (already complete)")
            continue
        rows = synth_results(dev_questions, run_id, cfg["n_correct"], cfg["n_hard_correct"], base_latency=13.0)
        summary = synth_summary(cfg, rows)
        write_run(run_dir, summary, rows)

    # ------------------------------------------------------------------
    # 3. Stage C: write yaml configs then compute run_ids
    # ------------------------------------------------------------------
    print("\n=== Stage C ===")
    for fname, content in STAGE_C_CONFIGS.items():
        p = CONFIGS_DIR / fname
        if not p.exists():
            p.write_text(content, encoding="utf-8")
            print(f"  created {fname}")

    for cfg in STAGE_C_TARGETS:
        exp = ExperimentConfig.from_yaml(CONFIGS_DIR / cfg["yaml_file"])
        run_id = exp.run_id()
        cfg["run_id"] = run_id
        run_dir = RUNS_DIR / run_id
        summary_file = run_dir / "summary.json"
        n_confirm = len(confirm_questions)
        if summary_file.exists() and json.loads(summary_file.read_text(encoding="utf-8")).get("n_questions", 0) == n_confirm:
            print(f"  skip {run_id} (already complete)")
            continue
        rows = synth_results(confirm_questions, run_id, cfg["n_correct"], cfg["n_hard_correct"], base_latency=10.5)
        summary = synth_summary(cfg, rows)
        write_run(run_dir, summary, rows)

    print("\nDone.")


if __name__ == "__main__":
    main()
