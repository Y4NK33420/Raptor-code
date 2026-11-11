"""Repo-QA evaluation for raptor.code: LLM-judge rubric over architecture questions.

    python scripts/code_qa.py refs      # draft reference answers from the gold files (LLM)
    python scripts/code_qa.py run       # answer with raptor (tree) and a flat symbol-RAG baseline
    python scripts/code_qa.py report

Reference answers are LLM-drafted from the full source of the gold files and are flagged
``needs_human_review``; scores are therefore indicative, not ground truth.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "qa"
REPOS = {
    "raptor": (ROOT, ROOT / "data" / "indexes" / "raptor"),
    "requests": (ROOT / "data" / "repos" / "requests", ROOT / "data" / "indexes" / "requests"),
    "flask": (ROOT / "data" / "repos" / "flask", ROOT / "data" / "indexes" / "flask"),
}
QUESTIONS = Path(__file__).with_name("code_qa_questions.json")
FLAT = {
    "kinds": ("symbol",),
    "bm25": False,
    "graph_expansion": False,
    "descend": False,
    "routing": False,
}


def load_index(repo: str):
    from raptor.code import CodeIndex

    return CodeIndex.load(REPOS[repo][1])


def _gold_source(repo: str, gold: list[str]) -> str:
    base = REPOS[repo][0]
    parts = []
    for g in gold:
        p = base / g
        files = sorted(p.rglob("*.py")) if p.is_dir() else [p]
        for f in files:
            body = f.read_text(encoding="utf-8")[:30000]
            parts.append(f"### {f.relative_to(base).as_posix()}\n{body}")
    return "\n\n".join(parts)


def refs() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "questions.json"
    items = json.loads(out.read_text()) if out.exists() else json.loads(QUESTIONS.read_text())
    models: dict = {}
    for it in items:
        if it.get("reference"):
            continue
        idx = models.setdefault(it["repo"], load_index(it["repo"]))
        prompt = (
            "Write a reference answer (4-8 sentences) to the question using ONLY the source "
            "below. Name the specific functions/classes involved and state how they work.\n\n"
            f"Question: {it['q']}\n\nSource:\n{_gold_source(it['repo'], it['gold'])}\n\nAnswer:"
        )
        it["reference"] = idx.models.qa.answer_question(prompt, "Write the reference answer.").strip()
        it["needs_human_review"] = True
        out.write_text(json.dumps(items, indent=1), encoding="utf-8")


def run() -> None:
    from raptor.code.bench import judge_answer

    items = json.loads((DATA / "questions.json").read_text())
    out = DATA / "results.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    indexes: dict = {}
    for n, it in enumerate(items):
        if str(n) in res:
            continue
        idx = indexes.setdefault(it["repo"], load_index(it["repo"]))
        row: dict = {"repo": it["repo"]}
        tree_text = idx.ask(it["q"], k=10).answer
        ctx, _ = idx._context(idx.search(it["q"], 10, **FLAT))
        flat_text = idx.models.qa.answer_question(ctx, it["q"])
        for name, text in (("tree", tree_text), ("flat", flat_text)):
            j = judge_answer(idx.models.qa, it["q"], it["reference"], text)
            row[name] = {k: j[k] for k in ("correctness", "completeness", "grounding", "comment")}
            row[name]["answer"] = text
            row[name]["cites_path_line"] = bool(re.search(r"\[[^\]\s]+:\d+", text))
        res[str(n)] = row
        out.write_text(json.dumps(res, indent=1), encoding="utf-8")
        logging.info("qa %d/%d", n + 1, len(items))


def report() -> str:
    res = json.loads((DATA / "results.json").read_text())
    rows = list(res.values())
    lines = [
        "| Repo | n | System | Correctness | Completeness | Grounding | Cites path:line |",
        "|---|---|---|---|---|---|---|",
    ]
    for repo in [*sorted({r["repo"] for r in rows}), "all"]:
        sub = [r for r in rows if repo == "all" or r["repo"] == repo]
        for system in ("tree", "flat"):
            m = [
                np.nanmean([r[system][k] for r in sub])
                for k in ("correctness", "completeness", "grounding")
            ]
            c = np.mean([r[system]["cites_path_line"] for r in sub])
            cells = " | ".join(f"{x:.2f}" for x in m)
            lines.append(f"| {repo} | {len(sub)} | {system} | {cells} | {c:.0%} |")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for noisy in ("httpx", "httpx2", "google_genai", "numba", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "report":
        print(report())
    else:
        {"refs": refs, "run": run}[cmd]()
