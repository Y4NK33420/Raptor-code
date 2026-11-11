"""Run the 12 raptor-repo QA questions against our local index and score them."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / ".cache" / "raptor" / "indexes" / "raptor-code"
QUESTIONS_FILE = Path(__file__).with_name("code_qa_questions.json")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(ROOT))

from raptor.code import CodeIndex  # noqa: E402

print(f"Loading index from {INDEX} ...", flush=True)
idx = CodeIndex.load(INDEX)
stats = idx.stats()
total_nodes = sum(stats["nodes"].values())
print(f"Index: {total_nodes} nodes, {stats['layers']} layers\n", flush=True)

questions = json.loads(QUESTIONS_FILE.read_text())
raptor_qs = [q for q in questions if q["repo"] == "raptor"]
print(f"Running {len(raptor_qs)} raptor questions\n{'='*60}\n", flush=True)

results = []
for i, q in enumerate(raptor_qs, 1):
    question = q["q"]
    gold_files = q["gold"]
    print(f"Q{i}: {question}", flush=True)
    ans = idx.ask(question)
    citations = [c for c in (ans.citations or []) if c]
    # Check if any gold file appears in citations
    hit = any(any(g in str(c) for g in gold_files) for c in citations)
    print(f"  Citations: {citations[:3]}")
    print(f"  Gold: {gold_files}  → Hit: {'✓' if hit else '✗'}")
    print(f"  Answer snippet: {ans.answer[:200]}\n", flush=True)
    results.append({
        "q": question,
        "gold": gold_files,
        "citations": [str(c.path) + f":{c.start_line}-{c.end_line}" for c in citations],
        "hit": hit,
        "answer": ans.answer,
    })

hits = sum(r["hit"] for r in results)
print(f"\n{'='*60}")
print(f"Citation recall (gold file in top citations): {hits}/{len(results)} = {hits/len(results):.1%}")

out = ROOT / "data" / "qa" / "results.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2))
print(f"Results saved to {out}")
