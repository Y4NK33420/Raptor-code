"""SWE-bench localization benchmark for raptor.code (retrieval only, no reader calls).

Usage::

    python scripts/code_localization.py fetch      # download instances, clone repos
    python scripts/code_localization.py run        # build indexes and score every baseline
    python scripts/code_localization.py report     # print the markdown tables

Everything lives under ``data/swebench/`` (gitignored). Model calls go through the disk
cache, so re-running is free once every index is built.
"""

from __future__ import annotations

import io
import json
import logging
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "swebench"
REPOS = ROOT / "data" / "repos"

# All 13 SWE-bench Lite instances from flask / requests / seaborn, topped up with 27
# instances of the same repos from the full SWE-bench test split, chosen in clusters of
# nearby base commits so most file summaries and embeddings are shared via the cache.
LITE = [
    "pallets__flask-4045", "pallets__flask-4992", "pallets__flask-5063",
    "psf__requests-863", "psf__requests-1963", "psf__requests-2148", "psf__requests-2317",
    "psf__requests-2674", "psf__requests-3362",
    "mwaskom__seaborn-2848", "mwaskom__seaborn-3010", "mwaskom__seaborn-3190",
    "mwaskom__seaborn-3407",
]  # fmt: skip
EXTRA = [
    "pallets__flask-4074", "pallets__flask-4160", "pallets__flask-4169",
    "pallets__flask-4935", "pallets__flask-5014",
    "psf__requests-1657", "psf__requests-1689", "psf__requests-1713", "psf__requests-1724",
    "psf__requests-1733", "psf__requests-1766", "psf__requests-1768", "psf__requests-1776",
    "psf__requests-1888", "psf__requests-1921", "psf__requests-1962", "psf__requests-2153",
    "mwaskom__seaborn-2846", "mwaskom__seaborn-2853", "mwaskom__seaborn-2946",
    "mwaskom__seaborn-2979", "mwaskom__seaborn-2996", "mwaskom__seaborn-3069",
    "mwaskom__seaborn-3180", "mwaskom__seaborn-3187", "mwaskom__seaborn-3202",
    "mwaskom__seaborn-3216",
]  # fmt: skip
SELECTED = LITE + EXTRA

EXCLUDE = [
    "tests/", "test/", "testing/", "docs/", "doc/", "examples/", "example/", "ci/",
    "test_*.py", "*_test.py", "conftest.py", "requests/packages/",
]  # fmt: skip

VARIANTS: list[tuple[str, str, bool, bool]] = [
    # (label, variant, bm25, graph)
    ("flat line-window", "lines", False, False),
    ("flat line-window + BM25", "lines", True, False),
    ("flat symbol", "symbols", False, False),
    ("flat symbol + BM25", "symbols", True, False),
    ("flat symbol + graph", "symbols", False, True),
    ("flat symbol + BM25 + graph", "symbols", True, True),
    ("structural tree", "structural", False, False),
    ("structural + BM25", "structural", True, False),
    ("structural + graph", "structural", False, True),
    ("structural + BM25 + graph", "structural", True, True),
    ("structural + semantic", "full", False, False),
    ("structural + semantic + BM25", "full", True, False),
    ("structural + semantic + graph", "full", False, True),
    ("structural + semantic + BM25 + graph", "full", True, True),
]

log = logging.getLogger("localization")


def _git(*args: str, cwd: Path | None = None) -> bytes:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True).stdout


def fetch() -> None:
    from datasets import load_dataset

    DATA.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name in ("princeton-nlp/SWE-bench_Lite", "princeton-nlp/SWE-bench"):
        for r in load_dataset(name, split="test"):
            if r["instance_id"] in SELECTED and r["instance_id"] not in rows:
                rows[r["instance_id"]] = {
                    k: r[k]
                    for k in ("instance_id", "repo", "base_commit", "patch", "problem_statement",
                              "version", "created_at")
                }  # fmt: skip
                rows[r["instance_id"]]["lite"] = r["instance_id"] in LITE
    missing = set(SELECTED) - rows.keys()
    if missing:
        raise SystemExit(f"instances not found: {sorted(missing)}")
    (DATA / "instances.json").write_text(
        json.dumps([rows[i] for i in SELECTED], indent=1, default=str), encoding="utf-8"
    )
    REPOS.mkdir(parents=True, exist_ok=True)
    for repo in sorted({r["repo"] for r in rows.values()}):
        dest = REPOS / repo.split("/")[1]
        if not dest.exists():
            log.info("cloning %s", repo)
            _git("clone", "--quiet", f"https://github.com/{repo}.git", str(dest))
    log.info("fetched %d instances", len(rows))


def checkout(inst: dict) -> Path:
    """Export the repo at the instance's base commit (no .git) under data/swebench/src."""
    dest = DATA / "src" / inst["instance_id"]
    if (dest / ".done").exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    clone = REPOS / inst["repo"].split("/")[1]
    blob = _git("archive", "--format=tar", inst["base_commit"], cwd=clone)
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        tar.extractall(dest, filter="data")
    (dest / ".done").write_text(inst["base_commit"])
    return dest


def eval_config():
    from raptor.code import CodeConfig

    cfg = CodeConfig()
    cfg.ingest.exclude = list(EXCLUDE)
    cfg.ingest.use_git = False
    cfg.summary.depth = "file"
    # Local embeddings: ~40 indexes plus line windows would exhaust the free Gemini
    # embedding quota. Summaries still use Gemini (cached across nearby commits).
    cfg.models.embedding_provider = "local"
    cfg.models.embedding_model = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
    cfg.models.use_task_types = False
    return cfg


def gold_for(inst: dict, src: Path) -> tuple[set[str], set[str]]:
    from raptor.code.bench import gold_functions, patch_old_lines

    touched = patch_old_lines(inst["patch"])
    files = {p for p in touched if p.endswith(".py")}
    funcs: set[str] = set()
    for path in files:
        f = src / path
        if f.exists():
            funcs |= {f"{path}::{q}" for q in gold_functions(f.read_bytes(), path, touched[path])}
    return files, funcs


def run(limit: int | None = None) -> None:
    from raptor.code import CodeIndex
    from raptor.code.bench import LineWindowIndex, rank_from_index, score_ranking

    instances = json.loads((DATA / "instances.json").read_text(encoding="utf-8"))
    out_path = DATA / "results.json"
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    for n, inst in enumerate(instances[:limit]):
        iid = inst["instance_id"]
        if iid in results:
            continue
        t0 = time.time()
        src = checkout(inst)
        idx_dir = DATA / "idx" / iid
        if (idx_dir / "code.json").exists():
            index = CodeIndex.load(idx_dir)
        else:
            index = CodeIndex.build(src, eval_config())
            index.save(idx_dir)
            log.info("%s build stats: %s", iid, index.last_build_stats)
        emb = index.models.embedding
        lines = LineWindowIndex.build(index, emb)
        gold_files, gold_funcs = gold_for(inst, src)
        row: dict = {
            "repo": inst["repo"],
            "lite": inst["lite"],
            "gold_files": sorted(gold_files),
            "gold_funcs": sorted(gold_funcs),
            "indexed_files": len(index.files()),
            "scores": {},
        }
        query = inst["problem_statement"]
        for label, variant, bm25, graph in VARIANTS:
            if variant == "lines":
                files, funcs = lines.rank(query, emb, bm25=bm25)
            else:
                files, funcs = rank_from_index(index, query, variant, bm25=bm25, graph=graph)
            row["scores"][label] = {
                "file": score_ranking(files, gold_files),
                "func": score_ranking(funcs, gold_funcs) if gold_funcs else None,
                "top_files": files[:5],
            }
        results[iid] = row
        out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        log.info("[%d/%d] %s done in %.0fs", n + 1, len(instances), iid, time.time() - t0)


def report() -> str:
    from raptor.code.bench import mean_scores

    results = json.loads((DATA / "results.json").read_text(encoding="utf-8"))
    rows = list(results.values())
    lines = [
        f"Instances: {len(rows)} ({sum(r['lite'] for r in rows)} from SWE-bench Lite); "
        f"with function-level gold: {sum(bool(r['gold_funcs']) for r in rows)}.",
        "",
        "| Retriever | File R@1 | File R@5 | File R@10 | File MRR | Func R@1 | Func R@5 | Func R@10 | Func MRR |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, *_ in VARIANTS:
        f = mean_scores([r["scores"][label]["file"] for r in rows if label in r["scores"]])
        g = mean_scores(
            [r["scores"][label]["func"] for r in rows if r["scores"].get(label, {}).get("func")]
        )
        if not f:
            continue
        cells = [f"{f[k]:.3f}" for k in ("R@1", "R@5", "R@10", "MRR")]
        cells += [f"{g[k]:.3f}" for k in ("R@1", "R@5", "R@10", "MRR")] if g else ["-"] * 4
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Per repository (file R@5 / func R@5):")
    lines.append("")
    repos = sorted({r["repo"] for r in rows})
    lines.append("| Retriever | " + " | ".join(f"{r} (n={sum(x['repo'] == r for x in rows)})" for r in repos) + " |")
    lines.append("|---|" + "---|" * len(repos))
    for label, *_ in VARIANTS:
        cells = []
        for repo in repos:
            sub = [r for r in rows if r["repo"] == repo and label in r["scores"]]
            f = mean_scores([r["scores"][label]["file"] for r in sub])
            g = mean_scores([r["scores"][label]["func"] for r in sub if r["scores"][label]["func"]])
            cells.append(f"{f.get('R@5', 0):.2f} / {g.get('R@5', 0):.2f}" if f else "-")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "google_genai", "numba", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "fetch":
        fetch()
    elif cmd == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif cmd == "report":
        print(report())
    else:
        raise SystemExit(__doc__)
