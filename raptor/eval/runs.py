"""Read API over run artifacts (used by the report and by the web UI).

``artifacts/runs/<run_id>/`` holds ``config.yaml``, ``results.jsonl`` and
``summary.json``; see :mod:`raptor.eval.runner` for the field list.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .metrics import paired_comparison
from .runner import DEFAULT_RUNS_DIR, read_jsonl


def list_runs(root: str | Path = DEFAULT_RUNS_DIR) -> list[dict[str, Any]]:
    """Summaries of every finished run, oldest first."""
    root = Path(root)
    if not root.exists():
        return []
    out = []
    for f in sorted(root.glob("*/summary.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return sorted(out, key=lambda s: (s.get("created_at", ""), s.get("run_id", "")))


def load_run(
    run_id: str, root: str | Path = DEFAULT_RUNS_DIR
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``(summary, results rows)`` for one run."""
    run_dir = Path(root) / run_id
    summary_file = run_dir / "summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"no finished run {run_id!r} in {root}")
    return json.loads(summary_file.read_text(encoding="utf-8")), read_jsonl(
        run_dir / "results.jsonl"
    )


def find_run(name_or_id: str, root: str | Path = DEFAULT_RUNS_DIR) -> dict[str, Any]:
    """Look a run up by id, or by experiment name (the latest run with that name)."""
    runs = list_runs(root)
    for s in runs:
        if s["run_id"] == name_or_id:
            return s
    named = [s for s in runs if s.get("name") == name_or_id]
    if not named:
        raise KeyError(f"no run with id or name {name_or_id!r}")
    return named[-1]


def compare_runs(a_id: str, b_id: str, root: str | Path = DEFAULT_RUNS_DIR) -> dict[str, Any]:
    """Paired comparison of run ``b`` against run ``a`` on the questions both answered.

    Returns ``delta`` (acc(b) - acc(a)), ``mcnemar_p``, ``fixed`` / ``broken`` question
    ids (right in b but not a / the reverse), ``both_right``, ``both_wrong``, plus ``n``
    and ``delta_ci95``.
    """
    _, rows_a = load_run(a_id, root)
    _, rows_b = load_run(b_id, root)
    a = {r["question_id"]: bool(r["correct"]) for r in rows_a}
    b = {r["question_id"]: bool(r["correct"]) for r in rows_b}
    out = paired_comparison(a, b)
    out.update(a=a_id, b=b_id)
    return out
