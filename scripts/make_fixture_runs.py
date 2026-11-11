"""Generate the synthetic experiment runs in ``tests/fixtures/runs/``.

They follow the Phase 2 run-artifact contract (``config.yaml``, ``results.jsonl``,
``summary.json``) and use real QuALITY dev questions, but the predictions are SIMULATED:
each configuration gets a latent skill and every question a shared difficulty, so runs are
paired the way real ones are and McNemar tests behave realistically. They exist so the
Experiments page and the API tests have data before real sweeps are run.

    python scripts/make_fixture_runs.py   # needs data/quality (scripts/download_quality.py)
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from pathlib import Path

import yaml

from raptor.eval.datasets.quality import load_quality

OUT = Path("tests/fixtures/runs")
N_STORIES = 8

BASE = {
    "chunking": {"strategy": "sentence", "max_tokens": 100},
    "clustering": {"strategy": "raptor_gmm", "reduction_dimension": 10},
    "retrieval": {"mode": "collapsed", "top_k": 10, "max_tokens": 2000},
    "embedding": {"model": "gemini-embedding-001"},
}

RUNS = [
    # name, method, skill (logit), factor overrides, layer share, created
    ("flat-baseline", "flat", 0.15, {}, {"0": 1.0}, "2026-11-02T09:12:00Z"),
    (
        "raptor-baseline",
        "raptor",
        0.55,
        {},
        {"0": 0.58, "1": 0.31, "2": 0.11},
        "2026-11-02T10:40:00Z",
    ),
    (
        "chunk-250",
        "raptor",
        0.42,
        {"chunking": {"strategy": "sentence", "max_tokens": 250}},
        {"0": 0.66, "1": 0.34},
        "2026-11-03T14:05:00Z",
    ),
    (
        "contiguous-clusters",
        "raptor",
        0.85,
        {"clustering": {"strategy": "contiguous", "reduction_dimension": 10}},
        {"0": 0.49, "1": 0.36, "2": 0.15},
        "2026-11-04T08:30:00Z",
    ),
    (
        "traversal",
        "raptor",
        0.30,
        {"retrieval": {"mode": "traversal", "top_k": 4, "max_tokens": 2000}},
        {"0": 0.40, "1": 0.40, "2": 0.20},
        "2026-11-04T16:55:00Z",
    ),
    (
        "bge-small",
        "raptor",
        0.35,
        {"embedding": {"model": "local:bge-small-en-v1.5"}},
        {"0": 0.61, "1": 0.29, "2": 0.10},
        "2026-11-05T11:20:00Z",
    ),
    (
        "hybrid-bm25",
        "raptor",
        0.70,
        {"retrieval": {"mode": "collapsed", "top_k": 10, "max_tokens": 2000, "hybrid": "bm25+rrf"}},
        {"0": 0.63, "1": 0.27, "2": 0.10},
        "2026-11-06T13:45:00Z",
    ),
]


def fingerprint(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def describe(factor: str, v: dict) -> str:
    if factor == "chunking":
        return f"{v['strategy']}-{v['max_tokens']}"
    if factor == "clustering":
        return v["strategy"]
    if factor == "retrieval":
        return v["mode"] + ("+" + v["hybrid"] if v.get("hybrid") else "")
    return v["model"]


def main() -> None:
    stories = load_quality("dev", limit=N_STORIES)
    questions = [(s, q) for s in stories for q in s.questions]
    qrng = random.Random(7)
    difficulty = {
        q.question_id: qrng.gauss(0.9 if q.difficult else -0.4, 0.9) for _, q in questions
    }
    if OUT.exists():
        shutil.rmtree(OUT)
    for name, method, skill, over, share, created in RUNS:
        factors = {k: dict(v) for k, v in BASE.items()}
        for k, v in over.items():
            factors[k] = v
        cfg = {"name": name, "method": method, "split": "dev", "n_stories": N_STORIES, **factors}
        fp = fingerprint(cfg)
        run_id = f"{name}-{fp[:6]}"
        d = OUT / run_id
        d.mkdir(parents=True)
        (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        rng = random.Random(fp)
        rows = []
        for s, q in questions:
            p = 1 / (1 + math.exp(-(skill - difficulty[q.question_id] + rng.gauss(0, 0.35))))
            correct = rng.random() < p
            gold = q.gold_label or 1
            wrong = [i for i in range(1, len(q.options) + 1) if i != gold]
            predicted = gold if correct else rng.choice(wrong)
            retrieved = []
            next_id = {"0": 0, "1": 40, "2": 52}
            for _ in range(10):
                layer = rng.choices(list(share), weights=list(share.values()))[0]
                next_id[layer] += rng.randint(1, 3)
                retrieved.append(
                    {
                        "node_index": next_id[layer],
                        "layer": int(layer),
                        "score": round(rng.uniform(0.55, 0.86), 4),
                    }
                )
            retrieved.sort(key=lambda r: -r["score"])
            rows.append(
                {
                    "run_id": run_id,
                    "set_id": s.set_id,
                    "question_id": q.question_id,
                    "question": q.question,
                    "options": q.options,
                    "gold": gold,
                    "predicted": predicted,
                    "correct": correct,
                    "difficult": q.difficult,
                    "retrieved": retrieved,
                    "context_tokens": rng.randint(1500, 2000),
                    "latency_s": round(rng.uniform(0.6, 2.4), 3),
                    "rationale": (
                        f"[simulated] The context supports option {predicted}."
                        if correct
                        else f"[simulated] The retrieved passages point to option {predicted}."
                    ),
                }
            )
        with (d / "results.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        n = len(rows)
        k = sum(r["correct"] for r in rows)
        hard = [r for r in rows if r["difficult"]]
        flat = method == "flat"
        summary = {
            "run_id": run_id,
            "name": name,
            "created_at": created,
            "split": "dev",
            "method": method,
            "config_fingerprint": fp,
            "factors": {
                f: describe(f, factors[f])
                for f in ("chunking", "clustering", "retrieval", "embedding")
            },
            "n_questions": n,
            "accuracy": round(k / n, 4),
            "accuracy_hard": round(sum(r["correct"] for r in hard) / len(hard), 4)
            if hard
            else None,
            "ci95": wilson(k, n),
            "layer_share": share,
            "cost": {
                "llm_calls": n + (0 if flat else rng.randint(60, 110)),
                "embed_calls": N_STORIES * (1 if flat else 3) + n,
                "cache_hits": rng.randint(40, 300),
                "cache_misses": rng.randint(5, 60),
            },
            "build_seconds": round(rng.uniform(4, 9) if flat else rng.uniform(70, 160), 1),
            "query_seconds_mean": round(sum(r["latency_s"] for r in rows) / n, 3),
            "tree_stats": {
                "mean_layers": 0.0 if flat else round(rng.uniform(1.6, 2.4), 2),
                "mean_nodes": round(rng.uniform(28, 34), 1)
                if flat
                else round(rng.uniform(36, 48), 1),
                "mean_branching": 0.0 if flat else round(rng.uniform(4.2, 6.5), 2),
            },
            "simulated": True,
        }
        (d / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"{run_id}: acc={summary['accuracy']:.3f} n={n}")


if __name__ == "__main__":
    main()
