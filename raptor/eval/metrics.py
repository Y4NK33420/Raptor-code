"""Metrics for experiment runs: accuracy with CIs, paired tests, layer mix, tree stats."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from ..types import Tree

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 0


def accuracy(correct: Sequence[bool]) -> float:
    return float(np.mean(correct)) if len(correct) else float("nan")


def bootstrap_ci(
    correct: Sequence[bool],
    *,
    level: float = 0.95,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI for accuracy."""
    x = np.asarray(correct, dtype=np.float64)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(samples, len(x)))].mean(axis=1)
    alpha = (1 - level) / 2
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha)))


def paired_bootstrap_ci(
    a: Sequence[bool],
    b: Sequence[bool],
    *,
    level: float = 0.95,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """CI for ``acc(b) - acc(a)`` over the same questions (resampling question pairs)."""
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    if len(d) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(samples, len(d)))].mean(axis=1)
    alpha = (1 - level) / 2
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha)))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts ``b`` and ``c``."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_comparison(a: dict[str, bool], b: dict[str, bool]) -> dict[str, Any]:
    """Compare two runs question by question (only questions both answered).

    ``fixed`` are questions ``b`` gets right and ``a`` gets wrong; ``broken`` the reverse.
    """
    common = sorted(set(a) & set(b))
    fixed = [q for q in common if b[q] and not a[q]]
    broken = [q for q in common if a[q] and not b[q]]
    both_right = sum(1 for q in common if a[q] and b[q])
    both_wrong = sum(1 for q in common if not a[q] and not b[q])
    av = [a[q] for q in common]
    bv = [b[q] for q in common]
    delta = (accuracy(bv) - accuracy(av)) if common else float("nan")
    return {
        "n": len(common),
        "delta": delta,
        "delta_ci95": list(paired_bootstrap_ci(av, bv)),
        "mcnemar_p": mcnemar_exact(len(fixed), len(broken)),
        "fixed": fixed,
        "broken": broken,
        "both_right": both_right,
        "both_wrong": both_wrong,
    }


def layer_share(retrieved: Iterable[Iterable[dict[str, Any]]]) -> dict[str, float]:
    """Fraction of all retrieved nodes that came from each layer."""
    counts: dict[int, int] = {}
    for nodes in retrieved:
        for n in nodes:
            counts[int(n["layer"])] = counts.get(int(n["layer"]), 0) + 1
    total = sum(counts.values())
    return {str(k): counts[k] / total for k in sorted(counts)} if total else {}


def tree_stats(tree: Tree, token_count: Any = None) -> dict[str, float]:
    """Shape of one tree. ``layers`` counts the leaf layer too."""
    inner = [n for n in tree.all_nodes.values() if n.children]
    stats: dict[str, float] = {
        "layers": float(tree.num_layers + 1),
        "nodes": float(len(tree.all_nodes)),
        "leaves": float(len(tree.leaf_nodes)),
        "branching": float(np.mean([len(n.children) for n in inner])) if inner else 0.0,
    }
    if token_count is not None and inner:
        ratios = []
        for n in inner:
            child_tokens = sum(token_count(tree.all_nodes[c].text) for c in n.children)
            if child_tokens:
                ratios.append(token_count(n.text) / child_tokens)
        stats["compression"] = float(np.mean(ratios)) if ratios else 0.0
    return stats


def mean_tree_stats(per_tree: Sequence[dict[str, float]]) -> dict[str, float]:
    if not per_tree:
        return {"mean_layers": 0.0, "mean_nodes": 0.0, "mean_branching": 0.0}
    out = {
        "mean_layers": float(np.mean([t["layers"] for t in per_tree])),
        "mean_nodes": float(np.mean([t["nodes"] for t in per_tree])),
        "mean_branching": float(np.mean([t["branching"] for t in per_tree])),
        "mean_leaves": float(np.mean([t["leaves"] for t in per_tree])),
    }
    if all("compression" in t for t in per_tree):
        out["mean_compression"] = float(np.mean([t["compression"] for t in per_tree]))
    return out


def summarize_results(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy block of ``summary.json`` from ``results.jsonl`` rows."""
    correct = [bool(r["correct"]) for r in rows]
    hard = [bool(r["correct"]) for r in rows if r.get("difficult")]
    lat = [float(r.get("latency_s", 0.0)) for r in rows]
    return {
        "n_questions": len(rows),
        "accuracy": accuracy(correct),
        "accuracy_hard": accuracy(hard) if hard else None,
        "n_hard": len(hard),
        "ci95": list(bootstrap_ci(correct)),
        "layer_share": layer_share(r.get("retrieved", []) for r in rows),
        "query_seconds_mean": float(np.mean(lat)) if lat else 0.0,
        "context_tokens_mean": float(np.mean([r.get("context_tokens", 0) for r in rows]))
        if rows
        else 0.0,
    }
