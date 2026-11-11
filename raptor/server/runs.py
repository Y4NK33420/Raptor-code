"""Experiment runs written by the Phase 2 harness (``artifacts/runs/<run_id>/``).

Each run directory holds ``config.yaml``, ``results.jsonl`` (one row per question) and
``summary.json``. When :mod:`raptor.eval.runs` is importable its readers are used;
otherwise (or if they fail) the files are read directly with the same contract.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import random
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,160}$")

try:  # Phase 2 harness, when merged
    from ..eval import runs as _eval_runs  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - depends on Phase 2
    _eval_runs = None


def _plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _plain(v) for k, v in dataclasses.asdict(obj).items()}
    if hasattr(obj, "model_dump"):
        return _plain(obj.model_dump(mode="json"))
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


class RunStore:
    def __init__(self, dirs: list[Path]) -> None:
        self.dirs = dirs

    # ---------------------------------------------------------------- discover
    def _run_dirs(self) -> dict[str, Path]:
        out: dict[str, Path] = {}
        for root in self.dirs:
            if not root.is_dir():
                continue
            for d in sorted(root.iterdir()):
                if d.is_dir() and (d / "summary.json").is_file() and _SAFE.match(d.name):
                    out.setdefault(d.name, d)
        return out

    def _origin(self, path: Path) -> str:
        return "fixture" if "fixtures" in path.parts else "artifact"

    def list_all(self) -> list[dict[str, Any]]:
        rows = []
        for run_id, d in self._run_dirs().items():
            try:
                summary = self._summary(d)
            except Exception as exc:
                logger.warning("skipping unreadable run %s: %s", d, exc)
                continue
            summary.setdefault("run_id", run_id)
            summary["origin"] = self._origin(d)
            rows.append(summary)
        rows.sort(key=lambda s: str(s.get("created_at", "")), reverse=True)
        return rows

    def _summary(self, d: Path) -> dict[str, Any]:
        if _eval_runs is not None and hasattr(_eval_runs, "load_run"):
            try:
                run = _plain(_eval_runs.load_run(d))
                summary = run.get("summary", run) if isinstance(run, dict) else None
                if isinstance(summary, dict) and "accuracy" in summary:
                    return dict(summary)
            except Exception:  # pragma: no cover - fall back to direct reads
                logger.debug("raptor.eval.runs.load_run failed for %s", d, exc_info=True)
        return dict(json.loads((d / "summary.json").read_text(encoding="utf-8")))

    def path(self, run_id: str) -> Path:
        if not _SAFE.match(run_id):
            raise KeyError(run_id)
        dirs = self._run_dirs()
        if run_id not in dirs:
            raise KeyError(run_id)
        return dirs[run_id]

    # ------------------------------------------------------------------- load
    def results(self, run_id: str) -> list[dict[str, Any]]:
        d = self.path(run_id)
        f = d / "results.jsonl"
        if not f.is_file():
            return []
        with f.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def load(self, run_id: str) -> dict[str, Any]:
        d = self.path(run_id)
        summary = self._summary(d)
        summary.setdefault("run_id", run_id)
        summary["origin"] = self._origin(d)
        cfg_file = d / "config.yaml"
        config_text = cfg_file.read_text(encoding="utf-8") if cfg_file.is_file() else ""
        try:
            config = yaml.safe_load(config_text) or {}
        except yaml.YAMLError:
            config = {}
        return {
            "summary": summary,
            "config": config,
            "config_yaml": config_text,
            "results": self.results(run_id),
        }

    # ---------------------------------------------------------------- compare
    def compare(self, a: str, b: str) -> dict[str, Any]:
        ra = {(r["set_id"], r["question_id"]): r for r in self.results(a)}
        rb = {(r["set_id"], r["question_id"]): r for r in self.results(b)}
        keys = sorted(ra.keys() & rb.keys())
        if _eval_runs is not None and hasattr(_eval_runs, "compare_runs"):
            try:
                out = _plain(_eval_runs.compare_runs(self.path(a), self.path(b)))
                if isinstance(out, dict) and "mcnemar_p" in out:
                    out.setdefault("questions", _question_rows(keys, ra, rb))
                    out.update(a=a, b=b)
                    return out
            except Exception:  # pragma: no cover
                logger.debug("raptor.eval.runs.compare_runs failed", exc_info=True)
        return compare_results(a, b, keys, ra, rb)


def _question_rows(keys, ra, rb) -> list[dict[str, Any]]:
    rows = []
    for k in keys:
        x, y = ra[k], rb[k]
        status = (
            "both_right"
            if x["correct"] and y["correct"]
            else "fixed"
            if y["correct"]
            else "broken"
            if x["correct"]
            else "both_wrong"
        )
        rows.append(
            {
                "set_id": k[0],
                "question_id": k[1],
                "question": y.get("question") or x.get("question"),
                "options": y.get("options") or x.get("options"),
                "gold": y.get("gold"),
                "difficult": bool(y.get("difficult", x.get("difficult", False))),
                "a_predicted": x.get("predicted"),
                "b_predicted": y.get("predicted"),
                "a_rationale": x.get("rationale"),
                "b_rationale": y.get("rationale"),
                "a_layers": _layer_mix(x),
                "b_layers": _layer_mix(y),
                "status": status,
            }
        )
    return rows


def _layer_mix(row: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in row.get("retrieved") or []:
        k = str(r.get("layer", 0))
        out[k] = out.get(k, 0) + 1
    return out


def mcnemar_exact(fixed: int, broken: int) -> float:
    """Two-sided exact McNemar test (binomial on the discordant pairs)."""
    n = fixed + broken
    if n == 0:
        return 1.0
    k = min(fixed, broken)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_bootstrap_ci(
    a: list[int], b: list[int], n_boot: int = 2000, seed: int = 0
) -> list[float]:
    if not a:
        return [0.0, 0.0]
    rng = random.Random(seed)
    n = len(a)
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(b[i] - a[i] for i in idx) / n)
    deltas.sort()
    return [round(deltas[int(0.025 * n_boot)], 4), round(deltas[int(0.975 * n_boot) - 1], 4)]


def compare_results(a, b, keys, ra, rb) -> dict[str, Any]:
    rows = _question_rows(keys, ra, rb)
    counts = {
        s: sum(r["status"] == s for r in rows)
        for s in ("fixed", "broken", "both_right", "both_wrong")
    }
    xa = [int(bool(ra[k]["correct"])) for k in keys]
    xb = [int(bool(rb[k]["correct"])) for k in keys]
    n = len(keys)
    acc_a = sum(xa) / n if n else 0.0
    acc_b = sum(xb) / n if n else 0.0
    return {
        "a": a,
        "b": b,
        "n_paired": n,
        "accuracy_a": round(acc_a, 4),
        "accuracy_b": round(acc_b, 4),
        "delta": round(acc_b - acc_a, 4),
        "delta_ci95": paired_bootstrap_ci(xa, xb),
        "mcnemar_p": round(mcnemar_exact(counts["fixed"], counts["broken"]), 5),
        **counts,
        "questions": rows,
    }
